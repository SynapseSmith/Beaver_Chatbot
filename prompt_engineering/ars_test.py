from pydub import AudioSegment
from pydub.playback import play

def play_mp3(file_path):
    try:
        # MP3 파일 로드
        audio = AudioSegment.from_mp3(file_path)
        
        # 오디오 재생
        play(audio)
        print('재생')
    except Exception as e:
        print(f"Error playing audio: {e}")

# 사용 예시
play_mp3("/home/user06/beaver/output.mp3")  # "your_file.mp3"를 실제 파일 경로로 바꿔주세요.