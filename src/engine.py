import music21
import copy
from typing import List, Dict, Any

class FugueEngine:
    """바흐 카운터포인트 모방 대위법 규칙에 입각해 2성부 화성을 작곡하고 스코어를 구성하는 규칙 기반 생성 엔진입니다."""

    def __init__(self) -> None:
        """FugueEngine의 새 인스턴스를 초기화합니다."""
        pass

    def is_pitch_in_key(self, pitch: music21.pitch.Pitch, key: music21.key.Key) -> bool:
        """주어진 음높이가 특정한 조(Key)의 다이아토닉 음계에 속하는지 판별합니다.
        
        Args:
            pitch: 검사할 music21 Pitch 객체.
            key: 기준이 되는 music21 Key 객체.
            
        Returns:
            bool: 해당 조에 속하는 음이면 True, 아니면 False.
        """
        pc = pitch.pitchClass
        key_pcs = [p.pitchClass for p in key.getScale().getPitches()]
        return pc in key_pcs

    def analyze_subject(self, notes_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """입력된 딕셔너리 형태의 소프라노 선율 데이터로부터 music21 Stream을 구성하고 조성을 자동으로 분석합니다.
        
        Args:
            notes_data: 각 음표의 pitch, duration, offset 정보를 담은 딕셔너리 리스트.
            
        Returns:
            Dict[str, Any]: 생성된 music21 Stream 객체('stream')와 분석된 Key 객체('key').
        """
        s = music21.stream.Stream()
        if not notes_data:
            # 빈 선율 입력값 수신 시 다장조(C Major)로 안전한 기본 폴백
            key = music21.key.Key('C')
            return {"stream": s, "key": key}

        for nd in notes_data:
            n = music21.note.Note(midi=int(nd['pitch']))
            n.duration.quarterLength = nd['duration']
            s.insert(nd['offset'], n)
        key = s.analyze('key')
        return {"stream": s, "key": key}

    def generate_tonal_answer(self, subject_stream: music21.stream.Stream, key: music21.key.Key) -> music21.stream.Stream:
        """입력된 소프라노 주제 선율을 5도 높은 딸림조(Dominant)로 이조하여 정석적인 응답 선율(Answer)을 생성합니다.
        
        바흐 푸가 형식에 입각하여 토닉(Tonic) 음과 도미넌트(Dominant) 음의 치환 처리를 자동 조율합니다.
        
        Args:
            subject_stream: 소프라노 주제의 music21 Stream 객체.
            key: 분석된 현재 곡의 조성(Key) 객체.
            
        Returns:
            music21.stream.Stream: 생성된 응답 선율(Answer)을 수록한 Stream.
        """
        answer = music21.stream.Stream()
        # 딸림조(V)로 이동
        dom_key = key.transpose('P5')
        
        for n in subject_stream.recurse().notes:
            new_note = n.transpose('P5')
            # Tonal logic
            if n.pitch.pitchClass == key.getDominant().pitchClass:
                new_note = music21.note.Note(dom_key.tonic)
            elif n.pitch.pitchClass == key.tonic.pitchClass:
                new_note = music21.note.Note(dom_key.getDominant())
            
            # Use original duration
            new_note.duration.quarterLength = n.duration.quarterLength
            new_note.pitch.octave = n.transpose('P5').pitch.octave
            answer.insert(n.offset, new_note)
        return answer

    def generate_countersubject(self, subject_stream: music21.stream.Stream, key: music21.key.Key) -> music21.stream.Stream:
        """주제의 리듬을 보존하며 화성적으로 대응하는 대위주제를 만듭니다.
        
        Args:
            subject_stream: 소프라노 주제의 music21 Stream 객체.
            key: 분석된 현재 곡의 조성(Key) 객체.
            
        Returns:
            music21.stream.Stream: 대응하는 대위주제(Countersubject)를 수록한 Stream.
        """
        cs = music21.stream.Stream()
        for n in subject_stream.recurse().notes:
            # 3도 혹은 6도 아래
            cs_note = n.transpose('-M3')
            # 조성 보정
            if not self.is_pitch_in_key(cs_note.pitch, key):
                cs_note = music21.note.Note(key.getScale().nextPitch(cs_note.pitch))
                cs_note.pitch.octave = n.pitch.octave
            
            # 리듬 다양성을 위해 가끔 음표를 반으로 쪼개거나 건너뛰는 로직 추가 가능 (우선 리듬 유지)
            cs_note.duration.quarterLength = n.duration.quarterLength
            cs.insert(n.offset, cs_note)
        return cs

    def assemble_hybrid_score(self, subject_notes: List[Dict[str, float]], neural_resp: List[Dict[str, float]]) -> Dict[str, Any]:
        """신경망이 생성한 대위선율(VOICE 2)을 입력된 주제(VOICE 1)와 결합하여 2성부 곡을 구성합니다.
        
        Args:
            subject_notes: 원본 소프라노 주제 음표 딕셔너리 리스트.
            neural_resp: 신경망에서 보정하여 생성한 대응 성부 음표 딕셔너리 리스트.
            
        Returns:
            Dict[str, Any]: key(조성 문자열), part1(원선율), part2(대선율), duration_total(총 연주시간) 정보.
        """
        # 기본 분석 (조성 등)
        analysis = self.analyze_subject(subject_notes)
        key = analysis['key']
        
        # Part 1: Original Subject
        part1 = subject_notes
        
        # Part 2: Neural Response
        # 신경망 결과에 오프셋 보정이 필요할 수 있으나, 우선 그대로 사용
        part2 = neural_resp
        
        # 전체 길이 계산
        max_t1 = max([n['offset'] + n['duration'] for n in part1]) if part1 else 0
        max_t2 = max([n['offset'] + n['duration'] for n in part2]) if part2 else 0
        total_len = max(max_t1, max_t2)

        return {
            "key": str(key),
            "part1": part1,
            "part2": part2,
            "duration_total": total_len
        }

    def compose_full_piece(self, notes_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """입력된 단선율로부터 제시부, 발전부, 종결부 구조를 지닌 완전한 2성부 대위법 곡을 자동 작곡합니다.
        
        Args:
            notes_data: 원본 단선율 음표 딕셔너리 리스트.
            
        Returns:
            Dict[str, Any]: 조성, 각 성부별 음표 시퀀스 및 총 길이 메타데이터.
        """
        analysis = self.analyze_subject(notes_data)
        subject = analysis['stream']
        key = analysis['key']
        sub_len = subject.highestTime
        if sub_len == 0: sub_len = 4.0
        
        part1 = music21.stream.Part()
        part2 = music21.stream.Part()
        
        # 1. 제시부 (0 ~ sub_len*2)
        for n in subject.recurse().notes:
            part1.insert(n.offset, copy.deepcopy(n))
            
        answer = self.generate_tonal_answer(subject, key)
        cs = self.generate_countersubject(subject, key)
        
        for n in answer.recurse().notes:
            part2.insert(n.offset + sub_len, copy.deepcopy(n))
        for n in cs.recurse().notes:
            part1.insert(n.offset + sub_len, copy.deepcopy(n))
            
        # 2. 발전부 (Episode) (sub_len*2 ~ sub_len*4)
        # 다양한 조성이동 (m3, P4)
        intervals = ['m3', 'P4']
        for i, interval in enumerate(intervals):
            offset = sub_len * 2 + (i * sub_len)
            seq_s = subject.transpose(interval)
            seq_cs = cs.transpose(interval)
            
            for n in seq_s.recurse().notes:
                if not self.is_pitch_in_key(n.pitch, key):
                    n.pitch = key.getScale().nextPitch(n.pitch)
                part2.insert(n.offset + offset, copy.deepcopy(n))
            for n in seq_cs.recurse().notes:
                if not self.is_pitch_in_key(n.pitch, key):
                    n.pitch = key.getScale().nextPitch(n.pitch)
                part1.insert(n.offset + offset, copy.deepcopy(n))

        # 3. 종결부 (sub_len*4 ~ )
        final_offset = sub_len * 4
        for n in subject.recurse().notes:
            part1.insert(n.offset + final_offset, copy.deepcopy(n))
        
        # 페달 포인트 (Tonic)
        tonic_base = music21.note.Note(key.tonic)
        tonic_base.pitch.octave = 3
        tonic_base.duration.quarterLength = sub_len
        part2.insert(final_offset, tonic_base)
            
        # 결과 추출 및 Offset 정렬
        p1_notes = []
        for n in part1.flatten().recurse().notes:
            p1_notes.append({"pitch": n.pitch.midi, "duration": n.duration.quarterLength, "offset": n.offset})
        p2_notes = []
        for n in part2.flatten().recurse().notes:
            p2_notes.append({"pitch": n.pitch.midi, "duration": n.duration.quarterLength, "offset": n.offset})
        
        return {
            "key": str(key),
            "part1": p1_notes,
            "part2": p2_notes,
            "duration_total": part1.highestTime
        }
