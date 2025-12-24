#!/usr/bin/env python3
import os
import sys
import requests
import json
import time
from pathlib import Path
_D = Path(__file__).resolve().parent
_R = _D.parent
pp = str(_R)
if pp not in sys.path:
    sys.path.insert(0, pp)

from videoautoclip.settings import init_settings_db, load_settings_into_env
# init_settings_db()
load_settings_into_env()
os.environ['FFMPEG_BINARY'] = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
from videoautoclip.minio_client import upload_file
def upload_file_to_minio(file_path):
    """
    上传文件到自定义MinIO服务器并返回URL
    
    Args:
        file_path: 本地文件路径
        
    Returns:
        文件在MinIO上的访问URL
    """
    try:
        # 导入MinIO客户端
        
        
        # 获取文件名
        import os
        file_name = os.path.basename(file_path)
        
        # 读取文件内容
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 上传到MinIO并获取URL
        file_url = upload_file(file_data, file_name)
        return file_url
        
    except Exception as e:
        print(f"上传文件到MinIO失败: {e}")
        raise
def _compress_audio_if_needed( audio_path, output_dir):
    """
    检查音频文件大小并在必要时进行压缩
    :param audio_path: 音频文件路径
    :param output_dir: 输出目录
    :return: 处理后的音频文件路径
    """
    import os
    
    # 检查文件大小
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"原始音频文件大小: {file_size_mb:.2f} MB")
    
    # 使用pydub处理音频时长
    try:
        from pydub import AudioSegment
        
        # 加载音频文件
        audio = AudioSegment.from_file(audio_path)
        
        # # 检查音频时长，如果超过3分钟(180秒)，则截取前3分钟
        # duration_seconds = len(audio) / 1000.0  # pydub以毫秒为单位
        # print(f"原始音频时长: {duration_seconds:.2f} 秒")
        
        # if duration_seconds > 180:  # 3分钟 = 180秒
        #     print("音频超过3分钟，正在截取前3分钟...")
        #     audio = audio[:180*1000]  # 截取前3分钟(180秒 = 180*1000毫秒)
        
        # # 降低音质以减小文件大小
        # # 1. 降低采样率到16kHz (如果原来是更高的话)
        # if audio.frame_rate > 16000:
        #     audio = audio.set_frame_rate(16000)
        
        # 2. 设置为单声道
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 3. 降低比特率
        compressed_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_compressed.wav"
        compressed_path = os.path.join(output_dir, compressed_filename)
        
        # 导出压缩后的音频
        audio.export(compressed_path, format="wav", bitrate="128k")
        
        # 检查压缩后的文件大小
        compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
        print(f"压缩后音频文件大小: {compressed_size_mb:.2f} MB")
        
        # 如果压缩后还是太大，则进一步降低质量
        if compressed_size_mb > 10:
            print("进一步压缩音频...")
            audio = audio.set_frame_rate(12000)  # 进一步降低采样率
            more_compressed_path = os.path.splitext(compressed_path)[0] + "_more.wav"
            audio.export(more_compressed_path, format="wav", bitrate="64k")
            final_size_mb = os.path.getsize(more_compressed_path) / (1024 * 1024)
            print(f"进一步压缩后音频文件大小: {final_size_mb:.2f} MB")
            return more_compressed_path
            
        return compressed_path
    except ImportError:
        print("无法导入pydub库，尝试使用librosa进行基本处理")
        # 使用librosa加载并重采样
        import librosa
        import soundfile as sf
        
        # 加载音频
        data, sr = librosa.load(audio_path, sr=None)
        
        # 检查音频时长，如果超过3分钟(180秒)，则截取前3分钟
        duration_seconds = len(data) / sr
        print(f"原始音频时长: {duration_seconds:.2f} 秒")
        
        if duration_seconds > 180:  # 3分钟 = 180秒
            print("音频超过3分钟，正在截取前3分钟...")
            # 计算前3分钟的样本数
            max_samples = 180 * sr
            data = data[:max_samples]
        
        # 重采样到16kHz
        if sr > 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # 确保是单声道
        if len(data.shape) > 1:
            data = data[:, 0]  # 取第一个声道
        
        # 保存压缩后的文件
        compressed_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_compressed.wav"
        compressed_path = os.path.join(output_dir, compressed_filename)
        sf.write(compressed_path, data, sr)
        
        return compressed_path
    except Exception as e:
        print(f"音频处理失败: {e}")
        # 如果处理失败，仍然使用原始文件（可能会失败）
        return audio_path

def qwen_asr_flash_async_recog(audio_input, language="zh", output_dir=None):
    """
    使用异步方式调用通义千问3-ASR-Flash进行音频识别
    :param audio_input: 音频输入，可以是文件路径或者是(gradio音频对象)
    :param language: 语言代码
    :param output_dir: 输出目录，用于保存临时文件
    :return: 识别结果(text, srt, state)
    """
    try:
        print(f"audio_input: {audio_input}")
        print(f"output_dir: {output_dir}")
        
        # 确保output_dir存在
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "."
            
     
        
        # 检查并压缩音频文件大小
        audio_path = _compress_audio_if_needed(audio_input, output_dir)
        print(f"压缩后的音频文件路径: {audio_path}")
        model_name = "qwen3-asr-flash-filetrans-2025-11-17"
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        #根据音频长度自动确定调用的模型，小于3分钟qwen3-asr-flash，大于3分钟qwen3-asr-flash-filetrans
        # import soundfile as sf
        # duration = sf.info(audio_path).duration
        # model_name = "qwen3-asr-flash"
        # #if duration < 180 else "qwen3-asr-flash-filetrans"
        # print(f"使用模型: {model_name}")
        
            #上传临时文件
        oss_url=upload_file_to_minio(audio_path)
        # oss_url=self.upload_file_and_get_url(os.environ.get("DASHSCOPE_API_KEY"), model_name, audio_path)
        print(f"oss_url: {oss_url}")
        # # 获取DashScope API密钥
        
        # if not dashscope_api_key:
        #     import dashscope
        #     dashscope_api_key = dashscope.api_key
            
        # if not dashscope_api_key:
        #     raise ValueError("未设置DASHSCOPE_API_KEY")
        
        # 提交异步转录任务
        url = os.environ.get("ASR_URL", "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription")
        print(f"dashscope_api_key: {dashscope_api_key}")
        headers = {
            "Authorization": f"Bearer {dashscope_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable"
        }
        
        # 直接使用返回的oss_url，不需要转换为HTTPS URL
        payload = {
            "model": model_name,
            "input": {
                "file_url": oss_url  # 直接使用oss_url
            },
            "parameters": {
                "channel_id": [0],
                "language": language,
                "enable_itn": False
            }
        }
        
        # 如果是本地文件，需要上传到可访问的位置
        # if not audio_path.startswith("http"):
        #     print("注意：对于异步ASR，音频文件需要可通过公网URL访问。当前实现可能需要额外的文件上传步骤。")
        print(f"payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            print(f"提交ASR异步任务失败: {response.text}")
            return "", "", {}
        
        task_info = response.json()
        task_id = task_info["output"]["task_id"]
        print(f"ASR异步任务已提交，任务ID: {task_id}")
        
        # 轮询获取任务结果
        base_task_url = os.environ.get("ASR_TASK_URL", "https://dashscope.aliyuncs.com/api/v1/tasks/")
        result_url = base_task_url.rstrip('/') + f"/{task_id}"
        result_headers = {
            "Authorization": f"Bearer {dashscope_api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }
        
        # 轮询间隔和最大尝试次数
        poll_interval = 5  # 秒
        max_attempts = 60   # 最多等待300秒(约5分钟)
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(poll_interval)
            result_response = requests.get(result_url, headers=result_headers)
            
            if result_response.status_code != 200:
                print(f"获取ASR任务结果失败: {result_response.text}")
                return "", "", {}
            
            result_data = result_response.json()
            task_status = result_data["output"]["task_status"]
            
            print(f"ASR任务状态: {task_status}")
            
            if task_status == "SUCCEEDED":
                # 任务成功完成
                transcription_url=result_data["output"]["result"]["transcription_url"]
                # 下载转录结果,使用requests直接获取json内容
                result_json = requests.get(transcription_url).json()
                print(f"result_json: {result_json}")

                # 任务成功完成，提取结果
                result_text = result_json["transcripts"][0]["text"]
                sentences = result_json["transcripts"][0].get("sentences", [])
                
                # 构造符合要求格式的返回数据
                state = {}
                state['audio_input'] = audio_path
                state['recog_res_raw'] = result_text
                # 将sentences转换为与recog方法兼容的格式
                formatted_sentences = []
                for i, sentence in enumerate(sentences):
                    # 添加timestamp字段以满足Text2SRT的要求
                    formatted_sentence = {
                        'start': int(sentence.get('begin_time', 0)),
                        'end': int(sentence.get('end_time', 0)),
                        'text': sentence.get('text', ''),
                        'emotion': sentence.get('emotion', ''),
                        'timestamp': [[int(sentence.get('begin_time', 0)), int(sentence.get('end_time', 0))]]
                    }
                    formatted_sentences.append(formatted_sentence)
                
                state['sentences'] = formatted_sentences
                # 修复：确保正确生成timestamp字段
                state['timestamp'] = [[int(s.get('begin_time', 0)), int(s.get('end_time', 0))] for s in sentences] if sentences else []
                
                # 生成SRT格式字幕
                from videoautoclip.utils.subtitle_utils import generate_srt
                res_srt = generate_srt(formatted_sentences)
                
                print(f"ASR识别完成，文本长度: {len(result_text)}")
                print(f"ASR识别结果: {result_text},{res_srt},{state}")
                return result_text, res_srt, state
                
            elif task_status == "FAILED":
                # 任务失败
                error_message = result_data.get("output", {}).get("message", "未知错误")
                print(f"ASR任务失败: {error_message}")
                return "", "", {}
            
            # 任务仍在运行中，继续轮询
            attempt += 1
        
        # 超时
        print("ASR任务执行超时")
        return "", "", {}
        
    except Exception as e:
        print(f"异步ASR调用出错: {e}")
        return "", "", {}

if __name__ == "__main__":
    asr_text, asr_srt, asr_state = qwen_asr_flash_async_recog("D:\\pythonwork\\VideoAutoCliped_compressed_more.wav","D:\\pythonwork\\VideoAutoClip\\data\\output")
    print(f"ASR识别结果: {asr_text}")
    print(f"ASR识别SRT: {asr_srt}")
    print(f"ASR识别状态: {asr_state}")