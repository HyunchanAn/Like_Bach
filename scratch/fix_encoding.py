import os

def fix_log_file():
    file_path = r"e:\Github\Like_Bach\development_log.txt"
    if not os.path.exists(file_path):
        print("Error: development_log.txt not found.")
        return
        
    with open(file_path, "rb") as f:
        content = f.read()
        
    # Find markers to segment the mixed encodings
    marker_19 = content.find(b"### 2026-04-19")
    marker_11 = content.find(b"[2026-05-11 17:35")
    
    if marker_19 == -1 or marker_11 == -1:
        print("Error: Could not find the required date markers in the file.")
        return
        
    # Segment 1: Start of file to just before April 19th (UTF-8)
    part1_bytes = content[:marker_19]
    part1 = part1_bytes.decode("utf-8", errors="replace")
    
    # Segment 2: April 19th to just before May 11th (CP949)
    part2_bytes = content[marker_19:marker_11]
    part2 = part2_bytes.decode("cp949", errors="replace")
    
    # Segment 3: May 11th to end of file (UTF-8)
    part3_bytes = content[marker_11:]
    part3 = part3_bytes.decode("utf-8", errors="replace")
    
    # Combine the healed segments
    full_healed_content = part1 + part2 + part3
    
    # Clean up minor formatting glitches in part 2 if present
    # Fix the "( \n eural_engine.py)" to "(neural_engine.py)"
    full_healed_content = full_healed_content.replace("(\nural_engine.py)", "(neural_engine.py)")
    full_healed_content = full_healed_content.replace("(\neural_engine.py)", "(neural_engine.py)")
    
    # Write the healed text back with clean UTF-8 encoding
    with open(file_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write(full_healed_content)
        
    print("Success: development_log.txt has been healed and saved as UTF-8.")

if __name__ == "__main__":
    fix_log_file()
