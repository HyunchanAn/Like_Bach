import music21
import copy
from typing import List, Dict

class FugueEngine:
    def __init__(self):
        pass

    def is_pitch_in_key(self, pitch, key):
        pc = pitch.pitchClass
        key_pcs = [p.pitchClass for p in key.getScale().getPitches()]
        return pc in key_pcs

    def analyze_subject(self, notes_data: List[Dict]) -> Dict:
        s = music21.stream.Stream()
        for nd in notes_data:
            n = music21.note.Note(midi=nd['pitch'])
            n.duration.quarterLength = nd['duration']
            s.insert(nd['offset'], n)
        key = s.analyze('key')
        return {"stream": s, "key": key}

    def generate_tonal_answer(self, subject_stream: music21.stream.Stream, key: music21.key.Key) -> music21.stream.Stream:
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
        """
        주제의 리듬을 보존하며 화성적으로 대응하는 대위주제를 만듭니다.
        """
        cs = music21.stream.Stream()
        for n in subject_stream.recurse().notes:
            # 3도 혹은 6도 아래
            cs_note = n.transpose('-M3')
            # 조성 보정
            if not self.is_pitch_in_key(cs_note.pitch, key):
                cs_note = music21.note.Note(key.getScale().next(cs_note.pitch))
                cs_note.pitch.octave = n.pitch.octave
            
            # 리듬 다양성을 위해 가끔 음표를 반으로 쪼개거나 건너뛰는 로직 추가 가능 (우선 리듬 유지)
            cs_note.duration.quarterLength = n.duration.quarterLength
            cs.insert(n.offset, cs_note)
        return cs

    def assemble_hybrid_score(self, subject_notes: List[Dict], neural_resp: List[Dict]) -> Dict:
        """
        신경망이 생성한 대위선율(VOICE 2)을 입력된 주제(VOICE 1)와 결합하여 2성부 곡을 구성합니다.
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

    def compose_full_piece(self, notes_data: List[Dict]) -> Dict:
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
                    n.pitch = key.getScale().next(n.pitch)
                part2.insert(n.offset + offset, copy.deepcopy(n))
            for n in seq_cs.recurse().notes:
                if not self.is_pitch_in_key(n.pitch, key):
                    n.pitch = key.getScale().next(n.pitch)
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
