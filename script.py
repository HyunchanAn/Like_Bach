import os

filepath = 'src/v5/neural_engine.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace _filter_logits
old_filter = '''    def _filter_logits(self, logits, voice, last_pitch, already_generated_notes, current_offset, current_seq):
        # 1. 복제본 생성
        masked_logits = logits.clone()
        
        # 성부별 정석 음역대 정의
        VOICE_RANGES = {
            1: (60, 81), # Soprano: C4 ~ A5
            2: (55, 76), # Alto: G3 ~ E5
            3: (48, 69), # Tenor: C3 ~ A4
            4: (40, 64)  # Bass: E2 ~ E4
        }
        
        # 2. 현재 성부의 음역대 가져오기
        v_min, v_max = VOICE_RANGES.get(voice, (40, 81))
        
        # 3. 현재 오프셋에 울리고 있는 다른 성부들의 (성부, 피치) 파악
        active_other_voices = []
        for note in already_generated_notes:
            if note['voice'] != voice:
                # 쉼표가 아닌 실음이 울리고 있는 구간 확인
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_other_voices.append((note['voice'], note['pitch']))
                    
        # 4. 각 토큰에 대해 검사
        for tid, p in self.token_pitches.items():
            # 4-1. 음역대 이탈 검사
            if p < v_min or p > v_max:
                masked_logits[0, tid] = -1e9
                continue
                
            # 4-2. 수평적 도약 제한
            # 1옥타브(12반음) 초과 도약 금지
            # 또한 감5도(6반음), 단7도(10반음), 장7도(11반음) 등의 불협화음 도약 방지
            if last_pitch is not None:
                interval = abs(p - last_pitch)
                if interval > 12: # 1옥타브 초과 도약 차단
                    masked_logits[0, tid] = -1e9
                    continue
                if interval in [6, 10, 11]: # 트라이톤, 단7도, 장7도 도약 제한
                    masked_logits[0, tid] = -1e9
                    continue
            
            # 4-3. 수직적 화성 협화 및 Voice Crossing 방지 (현재 오프셋에 다른 성부가 울리고 있을 때)
            for other_v, other_p in active_other_voices:
                harm_interval = abs(p - other_p)
                # 단2도(1), 장7도(11) 같은 거친 불협화음 원천 배제
                if harm_interval in [1, 11]:
                    masked_logits[0, tid] = -1e9
                    break
                # 증4도/감5도(트라이톤=6) 배제
                if harm_interval == 6:
                    masked_logits[0, tid] = -1e9
                    break
                # 성부 간 유니즌(음 겹침) 배제 (단, 옥타브는 허용하되 같은 피치 0은 차단)
                if harm_interval == 0:
                    masked_logits[0, tid] = -1e9
                    break
                    
                # Voice Crossing 방지: 성부 간의 수직적 순서 보장 (Soprano 1 > Alto 2 > Tenor 3 > Bass 4)
                if other_v < voice: # other_v가 더 높은 성부 (예: 1 < 2)
                    if p >= other_p: # 내가 더 높은 성부의 피치보다 같거나 높게 불면 Voice Crossing
                        masked_logits[0, tid] = -1e9
                        break
                elif other_v > voice: # other_v가 더 낮은 성부 (예: 4 > 3)
                    if p <= other_p: # 내가 더 낮은 성부의 피치보다 같거나 낮게 불면 Voice Crossing
                        masked_logits[0, tid] = -1e9
                        break
                                
        # 5. 과도한 쉼표 제약 및 듀레이션 제약
        rest_tid = self.tokenizer.stoi.get("[REST]", None)
        
        # 5-0. 32분음표 등 지나치게 짧은 음표 차단 (0.25 미만)
        for tid, dur in self.token_durations.items():
            if dur < 0.25:
                masked_logits[0, tid] = -1e9
                
        # 5-1. 현재 성부의 최근 생성 토큰들을 역순으로 추적
        voice_tokens = []
        i = len(current_seq) - 1
        while i >= 0:
            t = current_seq[i]
            if t == f"[VOICE_{voice}]":
                j = i + 1
                while j < len(current_seq):
                    nt = current_seq[j]
                    if nt.startswith("[VOICE_") or nt.startswith("[BAR_") or nt == "[FINAL]":
                        break
                    voice_tokens.append(nt)
                    j += 1
                if voice_tokens:
                    break
            i -= 1

        if len(voice_tokens) >= 1:
            last_t = voice_tokens[-1]
            if last_t == "[REST]":
                # REST 직후 -> 2.0박 이하의 듀레이션만 허용하고, 피치 및 쉼표 토큰 차단
                for tid, dur in self.token_durations.items():
                    if dur > 2.0:
                        masked_logits[0, tid] = -1e9
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("P"):
                # 피치 직후 -> 듀레이션만 허용
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("D"):
                # 듀레이션 직후 -> 다음 음(피치 또는 쉼표) 선택
                # 만약 직전 음표가 쉼표였다면 연속 쉼표 방지
                if len(voice_tokens) >= 2:
                    prev_t = voice_tokens[-2]
                    if prev_t == "[REST]":
                        if rest_tid is not None:
                            masked_logits[0, rest_tid] = -1e9
                            
        # 5-2. 다른 성부들이 이 오프셋에서 모두 쉬고 있다면, 이 성부는 쉴 수 없음 (침묵 구간 방지)
        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            active_voices_count = 0
            for note in already_generated_notes:
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_voices_count += 1
            if active_voices_count == 0 and len(already_generated_notes) > 0:
                masked_logits[0, rest_tid] = -1e9

        # 5-3. 쉼표 자체에 대한 페널티 부여 (쉼표가 과도하게 자주 선택되는 현상 방지)
        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            masked_logits[0, rest_tid] -= 3.5

        # 6. 세이프가드 (데드락 방지): 화성 규칙 등으로 인해 유효한 피치 후보가 극단적으로 적을 때, 쉼표가 강제되는 버그 방지
        num_active_pitches = sum([1 for tid in self.token_pitches if masked_logits[0, tid] > -1e8])
        if num_active_pitches < 3:
            # 쉼표를 제외한 피치 후보군이 고갈되면, 원래 logits로 복원하되 정석 음역대 이탈만 필터링
            masked_logits = logits.clone()
            for tid, p in self.token_pitches.items():
                if p < v_min or p > v_max:
                    masked_logits[0, tid] = -1e9
            # 복원 후에도 쉼표 페널티는 유지
            if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
                masked_logits[0, rest_tid] -= 3.5
                    
        return masked_logits'''

new_filter = '''    def _filter_logits(self, logits, voice, last_pitch, already_generated_notes, current_offset, current_seq):
        masked_logits = logits.clone()
        SOFT_PENALTY_VALUE = 3.0
        
        VOICE_RANGES = {
            1: (60, 81), 
            2: (55, 76), 
            3: (48, 69), 
            4: (40, 64)  
        }
        
        v_min, v_max = VOICE_RANGES.get(voice, (40, 81))
        
        active_other_voices = []
        for note in already_generated_notes:
            if note['voice'] != voice:
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_other_voices.append((note['voice'], note['pitch']))
                    
        counterpoint_violators = set()
        
        for tid, p in self.token_pitches.items():
            if p < v_min or p > v_max:
                masked_logits[0, tid] = -1e9
                continue
                
            is_violator = False
            
            if last_pitch is not None:
                interval = abs(p - last_pitch)
                if interval > 12: 
                    is_violator = True
                elif interval in [6, 10, 11]: 
                    is_violator = True
            
            for other_v, other_p in active_other_voices:
                harm_interval = abs(p - other_p)
                if harm_interval in [1, 11]:
                    is_violator = True
                    break
                if harm_interval == 6:
                    is_violator = True
                    break
                if harm_interval == 0:
                    is_violator = True
                    break
                    
                if other_v < voice: 
                    if p >= other_p: 
                        is_violator = True
                        break
                elif other_v > voice: 
                    if p <= other_p: 
                        is_violator = True
                        break
                        
            if is_violator:
                counterpoint_violators.add(tid)
                masked_logits[0, tid] = -1e9
                                
        rest_tid = self.tokenizer.stoi.get("[REST]", None)
        
        for tid, dur in self.token_durations.items():
            if dur < 0.25:
                masked_logits[0, tid] = -1e9
                
        voice_tokens = []
        i = len(current_seq) - 1
        while i >= 0:
            t = current_seq[i]
            if t == f"[VOICE_{voice}]":
                j = i + 1
                while j < len(current_seq):
                    nt = current_seq[j]
                    if nt.startswith("[VOICE_") or nt.startswith("[BAR_") or nt == "[FINAL]":
                        break
                    voice_tokens.append(nt)
                    j += 1
                if voice_tokens:
                    break
            i -= 1

        expecting_duration = False
        if len(voice_tokens) >= 1:
            last_t = voice_tokens[-1]
            if last_t == "[REST]":
                expecting_duration = True
                for tid, dur in self.token_durations.items():
                    if dur > 2.0:
                        masked_logits[0, tid] = -1e9
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("P"):
                expecting_duration = True
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("D"):
                if len(voice_tokens) >= 2:
                    prev_t = voice_tokens[-2]
                    if prev_t == "[REST]":
                        if rest_tid is not None:
                            masked_logits[0, rest_tid] = -1e9
                            
        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            active_voices_count = 0
            for note in already_generated_notes:
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_voices_count += 1
            if active_voices_count == 0 and len(already_generated_notes) > 0:
                masked_logits[0, rest_tid] = -1e9

        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            masked_logits[0, rest_tid] -= 3.5

        if not expecting_duration:
            num_active_pitches = sum([1 for tid in self.token_pitches if masked_logits[0, tid] > -1e8])
            if num_active_pitches < 3:
                for tid in counterpoint_violators:
                    masked_logits[0, tid] = logits[0, tid] - SOFT_PENALTY_VALUE
                    
        return masked_logits'''

content = content.replace(old_filter, new_filter)

# 2. Replace Exposition generation loop
old_exp_loop = '''                    # Voice has already entered -> AI generated Counterpoint
                    idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                    for _ in range(60):
                        logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                        
                        current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                        current_offset = voice_offsets[v]
                        
                        filtered_logits = self._filter_logits(
                            logits=logits[:, -1, :],
                            voice=v,
                            last_pitch=last_pitches[v],
                            already_generated_notes=current_notes,
                            current_offset=current_offset,
                            current_seq=current_seq
                        )
                        
                        probs = F.softmax(filtered_logits / temperature, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                        
                        if token.startswith("[VOICE_") or token.startswith("[BAR_") or token == "[FINAL]":
                            break
                            
                        if token.startswith("P"):
                            try:
                                p = int(token[1:])
                                last_pitches[v] = p
                            except Exception: pass
                            
                        idx = torch.cat((idx, idx_next), dim=1)
                        current_seq.append(token)'''

new_exp_loop = '''                    # Voice has already entered -> AI generated Counterpoint
                    idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                    import time
                    start_time = time.time()
                    invalid_strikes = 0
                    
                    for _ in range(120): # increased from 60 to 120 with timeout guards
                        if time.time() - start_time > 5.0:
                            print(f"Measure {m} Voice {v} Timeout!")
                            break
                            
                        logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                        
                        current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                        current_offset = voice_offsets[v]
                        
                        filtered_logits = self._filter_logits(
                            logits=logits[:, -1, :],
                            voice=v,
                            last_pitch=last_pitches[v],
                            already_generated_notes=current_notes,
                            current_offset=current_offset,
                            current_seq=current_seq
                        )
                        
                        probs = F.softmax(filtered_logits / temperature, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                        
                        if token == "[UNK]" or (token.startswith("D") and (len(current_seq) == 0 or not (current_seq[-1].startswith("P") or current_seq[-1] == "[REST]"))):
                            invalid_strikes += 1
                        else:
                            invalid_strikes = 0
                            
                        if token.startswith("[VOICE_") or token.startswith("[BAR_") or token == "[FINAL]" or invalid_strikes >= 5:
                            if invalid_strikes >= 5:
                                print(f"Measure {m} Voice {v} Max Invalid Strikes Reached!")
                            break
                            
                        if token.startswith("P"):
                            try:
                                p = int(token[1:])
                                last_pitches[v] = p
                            except Exception: pass
                            
                        idx = torch.cat((idx, idx_next), dim=1)
                        current_seq.append(token)
                        
                    # Fallback logic to pad measure if broken early
                    current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                    current_offset = voice_offsets.get(v, (m-1)*4.0)
                    target_offset = m * 4.0
                    if target_offset - current_offset > 0.05:
                        remaining = target_offset - current_offset
                        if remaining > 4.0: remaining = 4.0
                        current_seq.extend(["[REST]", f"D{round(remaining, 3)}"])'''

content = content.replace(old_exp_loop, new_exp_loop)

# 3. Replace Continuation generation loop
old_cont_loop = '''                # Generate notes for this voice until it tries to start a new voice or bar
                for _ in range(60): # Max 60 tokens per voice per measure
                    logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                    
                    current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                    current_offset = voice_offsets[v]
                    
                    filtered_logits = self._filter_logits(
                        logits=logits[:, -1, :],
                        voice=v,
                        last_pitch=last_pitches[v],
                        already_generated_notes=current_notes,
                        current_offset=current_offset,
                        current_seq=current_seq
                    )
                    
                    # Prevent generating structural tokens in the middle of a voice
                    for tid, tok in self.tokenizer.stoi.items():
                        if tok in ["[SUBJECT]", "[ANSWER]", "[EPISODE]", "[EPISODE_MODULATION]", "[TS_4/4]", "[KEY_C]"]:
                            filtered_logits[0, tid] = -1e9
                    
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                    
                    if token.startswith("[BAR_") or token.startswith("[VOICE_") or token == "[FINAL]":
                        break
                        
                    if token.startswith("P"):
                        try:
                            p = int(token[1:])
                            last_pitches[v] = p
                        except Exception: pass
                        
                    idx = torch.cat((idx, idx_next), dim=1)
                    current_seq.append(token)'''

new_cont_loop = '''                # Generate notes for this voice until it tries to start a new voice or bar
                import time
                start_time = time.time()
                invalid_strikes = 0
                
                for _ in range(120): # increased from 60 to 120 with timeout guards
                    if time.time() - start_time > 5.0:
                        print(f"Measure {m} Voice {v} Timeout!")
                        break
                        
                    logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                    
                    current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                    current_offset = voice_offsets[v]
                    
                    filtered_logits = self._filter_logits(
                        logits=logits[:, -1, :],
                        voice=v,
                        last_pitch=last_pitches[v],
                        already_generated_notes=current_notes,
                        current_offset=current_offset,
                        current_seq=current_seq
                    )
                    
                    # Prevent generating structural tokens in the middle of a voice
                    for tid, tok in self.tokenizer.stoi.items():
                        if tok in ["[SUBJECT]", "[ANSWER]", "[EPISODE]", "[EPISODE_MODULATION]", "[TS_4/4]", "[KEY_C]"]:
                            filtered_logits[0, tid] = -1e9
                    
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                    
                    if token == "[UNK]" or (token.startswith("D") and (len(current_seq) == 0 or not (current_seq[-1].startswith("P") or current_seq[-1] == "[REST]"))):
                        invalid_strikes += 1
                    else:
                        invalid_strikes = 0
                        
                    if token.startswith("[BAR_") or token.startswith("[VOICE_") or token == "[FINAL]" or invalid_strikes >= 5:
                        if invalid_strikes >= 5:
                            print(f"Measure {m} Voice {v} Max Invalid Strikes Reached!")
                        break
                        
                    if token.startswith("P"):
                        try:
                            p = int(token[1:])
                            last_pitches[v] = p
                        except Exception: pass
                        
                    idx = torch.cat((idx, idx_next), dim=1)
                    current_seq.append(token)
                    
                # Fallback logic to pad measure if broken early
                current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                current_offset = voice_offsets.get(v, (m-1)*4.0)
                target_offset = m * 4.0
                if target_offset - current_offset > 0.05:
                    remaining = target_offset - current_offset
                    if remaining > 4.0: remaining = 4.0
                    current_seq.extend(["[REST]", f"D{round(remaining, 3)}"])'''

content = content.replace(old_cont_loop, new_cont_loop)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Changes applied successfully!")
