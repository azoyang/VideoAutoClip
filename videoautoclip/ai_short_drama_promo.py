#!/usr/bin/env python3
# ai_short_drama_promo.py
import os
import re
import argparse
import json
import logging
import requests
from typing import List, Callable, Optional
from datetime import timedelta

# 可选：仅在需要时导入 moviepy（避免无图形环境报错）
try:
    from moviepy.editor import (
        VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip, afx
    )
    from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("moviepy 未安装，仅支持导出剪辑时间戳（无视频生成）")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FALLBACK_KEYWORDS = ["不要", "丢下", "对不起", "怀孕", "癌症", "替身", "永远", "再见", "恨你", "爱她"]

# ==================== SRT 解析 ====================
def srt_to_segments(srt_path: str):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 处理不同操作系统的换行符
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    full_text_parts = []
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        # 至少需要2行（序号行和时间轴行），文本可以为空
        if len(lines) < 2:
            continue
        try:
            # 文本从第3行开始，如果没有文本则为空字符串
            text = ' '.join(lines[2:]) if len(lines) > 2 else ""
            full_text_parts.append(text)
            time_line = lines[1]
            start_str, end_str = time_line.split(' --> ')
            start_sec = time_str_to_seconds(start_str)
            end_sec = time_str_to_seconds(end_str)
            segments.append({'start': start_sec, 'end': end_sec, 'text': text})
        except Exception as e:
            # 记录错误但继续处理其他块
            logging.debug(f"跳过无法解析的块: {lines}, 错误: {e}")
            continue
    return segments, '。'.join(full_text_parts)

def time_str_to_seconds(time_str: str) -> float:
    h, m, s_ms = time_str.replace(',', ':').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s_ms)

# ==================== LLM 接口 ====================
def create_llm_caller(model_name: str, base_url: Optional[str] = None) -> Callable[[str], str]:
    if model_name.startswith("gpt-"):
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        
        def call_openai(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            return resp.choices[0].message.content.strip()
        return call_openai

    elif model_name.startswith("qwen"):
        # 使用OpenAI兼容方式调用Qwen模型
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
            
        # Qwen模型使用DashScope的OpenAI兼容接口
        client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get("DASHSCOPE_BASE_HTTP_API_URL","https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        
        def call_qwen(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            return resp.choices[0].message.content.strip()
        return call_qwen

    elif base_url or model_name in ["llama3", "phi3", "mistral"]:
        url = (base_url or "http://localhost:11434") + "/api/generate"
        def call_ollama(prompt: str) -> str:
            resp = requests.post(
                url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 256}
                },
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        return call_ollama

    else:
        # 自建 API：假设 POST /v1/completions 返回 {"text": "..."}
        def call_custom(prompt: str) -> str:
            resp = requests.post(
                f"{base_url}/v1/completions",
                json={"prompt": prompt, "max_tokens": 256},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["text"].strip()
        return call_custom

# ==================== 关键词与钩子提取 ====================
def extract_keywords_with_llm(srt_text: str, llm_func: Callable[[str], str]) -> List[str]:
    prompt = f"""
你是一个短视频爆款内容分析师。请从以下短剧台词中，自动识别出最能引发观众情绪（如愤怒、心疼、震惊、好奇）的 **高能关键词或短语**（3–8个）。
要求：
- 必须是台词中实际出现的词或短句
- 优先选择包含冲突、反转、绝症、背叛、怀孕、替身、死亡、临终、打脸等元素的词
- 每个关键词不超过6个汉字
- 返回纯 JSON 列表，不要任何解释

台词内容：
{srt_text}
"""
    try:
        response = llm_func(prompt)
        print(f"LLM extract_keywords_with_llm：{response}")
        
        # 尝试直接解析响应
        try:
            keywords = json.loads(response)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取代码块中的JSON
            import re
            json_match = re.search(r'```(?:json)?\s*([^\]]*\])', response)
            if json_match:
                json_text = json_match.group(1)
                keywords = json.loads(json_text)
            else:
                # 如果仍然失败，使用备用关键词
                logging.warning(f"无法解析LLM响应为JSON: {response}")
                return FALLBACK_KEYWORDS
                
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            return [k.strip() for k in keywords if k.strip()]
    except Exception as e:
        logging.warning(f"LLM 关键词提取失败: {e}")
    return FALLBACK_KEYWORDS

def select_hook_with_llm(clips: List[tuple], srt_text: str, llm_func: Callable[[str], str]) -> Optional[tuple]:
    if not clips:
        return None
    candidates = "\n".join([f"{i+1}. {text}" for i, (_, _, text,_) in enumerate(clips[:5])])
    prompt = f"""
    你是一位拥有千万粉丝的抖音短剧导演，专精于“10秒钩子”设计。
    请从以下候选台词中，选出**唯一一句**最能让人**立刻停止滑动**的台词作为视频开头。

    【选择原则】（按优先级）：
    1. **情绪冲击最强**：包含极度愤怒、震惊、心碎、恐惧、绝望或狂喜
    2. **存在强烈反转或悬念**：如身份揭露（“你不是我亲生的！”）、秘密曝光（“孩子不是你的！”）、命运突变（“你中奖了，但要坐牢！”）
    3. **台词简短有力**：优先选择 ≤15 个汉字 的句子（越短越有冲击力）
    4. **包含具体冲突动作或结果**：如“下跪”、“打脸”、“离婚”、“报警”、“跳楼”、“转账一亿”，而非抽象抒情
    5. **禁止使用以下类型**：  
    - 礼貌用语、服务性语言（如“您的药”“请喝茶”）  
    - 无主语/无上下文的祈使句（如“别走”“等等”）  
    - 长度超过 20 字的句子

    【候选台词】：
    {candidates}

    【输出要求】：
    - 仅返回选中的台词原文（逐字，不加引号、编号或标点修正）
    - 不要任何解释、换行或额外字符
    - 若所有台词都平淡，选择情绪最浓烈的一句
    """
    try:
        hook_text = llm_func(prompt).strip().strip('"').strip("'")
        # 清理可能的代码块标记
        if hook_text.startswith("```") and hook_text.endswith("```"):
            hook_text = hook_text[3:-3].strip()
            
        for start, end, text,kw in clips:
            if hook_text in text or text.replace(" ", "") in hook_text.replace(" ", ""):
                return (start, end, text, kw)
        return (clips[0][0], clips[0][1], clips[0][2], "")
    except Exception as e:
        logging.warning(f"LLM 钩子选择失败: {e}")
        return (clips[0][0], clips[0][1], clips[0][2], "")

# ==================== 视频剪辑核心 ====================
def find_clips_by_keywords(segments: list, keywords: List[str], expand_sec: float = 1.0) -> List[tuple]:
    keyword_set = set(kw for kw in keywords if kw)
    clips = []
    for seg in segments:
        text = seg['text']
        for kw in keyword_set:
            if kw in text:
                start = max(0, seg['start'] - expand_sec)
                end = seg['end'] + expand_sec
                clips.append((start, end, text, kw))
                break
    return clips


def insert_images_into_video(video_clips: List, subtitle_items: List, clip_srt: str, 
                           image_paths_and_times: List[tuple], video_size: tuple, fps: float) -> tuple:
    """
    在视频的指定时间点插入图片
    
    Args:
        video_clips: 已有的视频片段列表
        subtitle_items: 已有的字幕项列表
        clip_srt: 已有的SRT字幕内容
        image_paths_and_times: 图片路径和插入时间的元组列表 [(path, start_time, duration), ...]
        video_size: 视频尺寸 (width, height)
        fps: 视频帧率
    
    Returns:
        tuple: (updated_video_clips, updated_subtitle_items, updated_clip_srt, updated_current_time)
    """
    if not image_paths_and_times:
        # 如果没有图片需要插入，直接返回原始数据
        # 计算当前总时长
        current_time = sum([clip.duration for clip in video_clips])
        return video_clips, subtitle_items, clip_srt, current_time
    
    try:
        from moviepy.editor import ImageClip
    except ImportError:
        logging.warning("moviepy.editor.ImageClip 无法导入，图片插入功能不可用")
        # 计算当前总时长
        current_time = sum([clip.duration for clip in video_clips])
        return video_clips, subtitle_items, clip_srt, current_time
    
    # 按时间排序图片插入点
    sorted_images = sorted(image_paths_and_times, key=lambda x: x[1])  # 按开始时间排序
    
    # 初始化结果列表
    result_video_clips = []
    result_subtitle_items = subtitle_items[:]  # 复制原始字幕项
    result_clip_srt = clip_srt
    current_time = 0.0
    
    # 当前处理的视频片段索引
    clip_index = 0
    total_video_duration = sum([clip.duration for clip in video_clips])
    
    # 处理每个图片插入点
    for img_path, img_start_time, img_duration in sorted_images:
        # 确保图片插入时间在合理范围内
        img_start_time = max(0, min(img_start_time, total_video_duration))
        
        # 添加在图片之前的所有视频片段
        while clip_index < len(video_clips):
            clip = video_clips[clip_index]
            clip_end_time = current_time + clip.duration
            
            # 如果当前片段结束时间在图片插入时间之前，则完整添加该片段
            if clip_end_time <= img_start_time:
                result_video_clips.append(clip)
                current_time = clip_end_time
                clip_index += 1
            else:
                # 需要在当前片段中插入图片
                # 计算在当前片段中插入点的相对位置
                relative_insert_time = img_start_time - current_time
                
                if relative_insert_time > 0:
                    # 在插入点之前分割片段
                    before_clip = video_clips[clip_index].subclip(0, relative_insert_time)
                    result_video_clips.append(before_clip)
                    current_time += before_clip.duration
                    
                    # 更新剩余片段供后续处理
                    remaining_clip = video_clips[clip_index].subclip(relative_insert_time)
                    video_clips[clip_index] = remaining_clip
                break
        
        # 插入图片片段
        try:
            if os.path.exists(img_path):
                def _make_image_clip(path: str, duration: float, size: tuple, fps: float):
                    p = os.path.normpath(str(path)).strip()
                    try:
                        from PIL import Image
                        img = Image.open(p).convert("RGB")
                        w, h = size
                        try:
                            resample = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample = Image.LANCZOS
                        img = img.resize((w, h), resample=resample)
                        import numpy as np
                        arr = np.array(img)
                        from moviepy.editor import ImageClip
                        return ImageClip(arr, duration=duration).set_fps(fps)
                    except Exception:
                        try:
                            import numpy as np
                            import cv2
                            data = np.fromfile(p, dtype=np.uint8)
                            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                            if img is None:
                                return None
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            w, h = size
                            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                            from moviepy.editor import ImageClip
                            return ImageClip(img, duration=duration).set_fps(fps)
                        except Exception:
                            return None

                image_clip = _make_image_clip(img_path, img_duration, video_size, fps)
                if image_clip:
                    result_video_clips.append(image_clip)
                    hours_start = int(current_time // 3600)
                    minutes_start = int((current_time % 3600) // 60)
                    seconds_start = int(current_time % 60)
                    milliseconds_start = int((current_time % 1) * 1000)
                    hours_end = int((current_time + img_duration) // 3600)
                    minutes_end = int(((current_time + img_duration) % 3600) // 60)
                    seconds_end = int((current_time + img_duration) % 60)
                    milliseconds_end = int(((current_time + img_duration) % 1) * 1000)
                    start_time_str = f"{hours_start:02d}:{minutes_start:02d}:{seconds_start:02d},{milliseconds_start:03d}"
                    end_time_str = f"{hours_end:02d}:{minutes_end:02d}:{seconds_end:02d},{milliseconds_end:03d}"
                    next_index = len(result_subtitle_items) + 1
                    result_subtitle_items.append(((current_time, current_time + img_duration), "[图片]"))
                    result_clip_srt += f"{next_index}\n{start_time_str} --> {end_time_str}\n[图片]\n\n"
                    current_time += img_duration
                else:
                    logging.warning(f"无法加载图片 {img_path}")
            else:
                logging.warning(f"图片文件不存在: {img_path}")
        except Exception as e:
            logging.warning(f"无法加载图片 {img_path}: {e}")
    
    # 添加剩余的所有视频片段
    while clip_index < len(video_clips):
        result_video_clips.append(video_clips[clip_index])
        clip_index += 1
    
    return result_video_clips, result_subtitle_items, result_clip_srt, current_time


def create_promo_video(
    video_path: str,
    srt_content: str,
    output_path: Optional[str] = None,
    llm_model: str = "qwen-plus",
    llm_base_url: Optional[str] = None,
    bgm_path: Optional[str] = None,
    font_path: str = './font/STHeitiMedium.ttc',
    font_size: int = 28,
    expand_sec: float = 10.0,
    max_clips: int = 5,
    cover_image_path: Optional[str] = None,
    image_inserts: Optional[List[tuple]] = None  # [(image_path, time, duration), ...]
):
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy 未安装，无法生成视频。请运行: pip install moviepy")
    
    # 如果未提供 output_path，则根据 video_path 自动生成
    if output_path is None or len(output_path) == 0:
        # 获取视频文件的目录和基本名称
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        # 移除扩展名并添加 _output.mp4 后缀
        base_name, _ = os.path.splitext(video_name)
        output_path = os.path.join(video_dir, f"{base_name}_output.mp4")
        
    # 1. 解析 SRT
    # 修改为同时支持文件路径和SRT内容字符串
    if srt_content.endswith('.srt') and os.path.isfile(srt_content):
        segments, srt_text = srt_to_segments(srt_content)
    else:
        # 直接解析SRT内容字符串
        srt_content_str = srt_content  # 这里实际上是SRT内容而非路径
        print(f"解析直接提供的 SRT 内容字符串={srt_content_str}")
        # 处理不同操作系统的换行符
        srt_content_str = srt_content_str.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', srt_content_str.strip())
        segments = []
        full_text_parts = []
        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            # 至少需要2行（序号行和时间轴行），文本可以为空
            if len(lines) < 2:
                continue

            try:
                # 文本从第3行开始，如果没有文本则为空字符串
                text = ' '.join(lines[2:]) if len(lines) > 2 else ""
                full_text_parts.append(text)

                # 解析时间轴
                time_line = lines[1]
                start_str, end_str = time_line.split(' --> ')

                # 定义时间字符串到秒的转换函数
                def time_str_to_seconds(time_str):
                    h, m, s = time_str.split(':')
                    s, ms = s.split(',')
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

                start_sec = time_str_to_seconds(start_str)
                end_sec = time_str_to_seconds(end_str)

                segments.append({'start': start_sec, 'end': end_sec, 'text': text})

            except Exception as e:
                # 记录错误但继续处理其他块
                print(f"跳过无法解析的块: {lines}, 错误: {e}")
                continue
        srt_text = '。'.join(full_text_parts)
        
    if not segments:
        raise ValueError("SRT 文件为空或格式错误")
    logging.info(f"✅ 解析 {len(segments)} 条字幕")

    # 2. 初始化 LLM
    llm_func = create_llm_caller(llm_model, llm_base_url)
    logging.info(f"🧠 使用 LLM: {llm_model}")

    # 3. 动态提取关键词
    keywords = extract_keywords_with_llm(srt_text, llm_func)
    print(f"🔑 高能关键词: {keywords}")

    # 4. 提取高能片段
    all_clips = find_clips_by_keywords(segments, keywords, expand_sec=expand_sec)
    if not all_clips:
        raise ValueError("未找到匹配关键词的片段")

    # 5. 选择黄金3秒钩子
    hook_clip = select_hook_with_llm(all_clips, srt_text, llm_func)
    print(f"🎣 黄金3秒: {hook_clip[2] if hook_clip else 'N/A'}")

    # 6. 构建剪辑列表（钩子 + 其他）
    video_clips = []
    subtitle_items = []
    current_time = 0.0

    # 加载主视频
    video = VideoFileClip(video_path)
    total_clips = [hook_clip] if hook_clip else []
    used_texts = {hook_clip[2]} if hook_clip else set()

    for clip in all_clips:
        if clip[2] not in used_texts and len(total_clips) < max_clips:
            total_clips.append(clip)
            used_texts.add(clip[2])

    # 7. 逐片段处理
    clip_srt = ""  # 收集所有片段的SRT内容
    current_time = 0.0  # 重置当前时间，如果添加封面需要额外时间
    
    # 如果提供了封面图片，则创建封面片段
    cover_clip = None
    if cover_image_path and os.path.exists(cover_image_path):
        try:
            def _make_image_clip(path: str, duration: float, size: tuple, fps: float):
                p = os.path.normpath(str(path)).strip()
                try:
                    from PIL import Image
                    img = Image.open(p).convert("RGB")
                    w, h = size
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.LANCZOS
                    img = img.resize((w, h), resample=resample)
                    import numpy as np
                    arr = np.array(img)
                    from moviepy.editor import ImageClip
                    return ImageClip(arr, duration=duration).set_fps(fps)
                except Exception:
                    try:
                        import numpy as np
                        import cv2
                        data = np.fromfile(p, dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if img is None:
                            return None
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        w, h = size
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                        from moviepy.editor import ImageClip
                        return ImageClip(img, duration=duration).set_fps(fps)
                    except Exception:
                        return None
            cover_clip = _make_image_clip(cover_image_path, 2.0, video.size, video.fps)
            if cover_clip is not None:
                current_time += 2.0
        except Exception as e:
            logging.warning(f"无法加载封面图片: {e}")
            cover_clip = None
    if cover_clip is None:
        try:
            from moviepy.editor import ImageClip
            frame = video.get_frame(0)
            cover_clip = ImageClip(frame, duration=2.0).set_fps(video.fps)
            current_time += 2.0
        except Exception:
            cover_clip = None

    # 处理主要视频片段
    video_clips = []
    subtitle_items = []
    
    # 如果有封面剪辑，则添加到视频片段列表开头
    if cover_clip is not None:
        video_clips.append(cover_clip)
        
        # 添加一个空字幕条目给封面（没有字幕）
        # 注意：封面不添加字幕，所以不更新subtitle_items
        
        # 封面的SRT条目
        start_time_str = "00:00:00,000"
        end_time_str = "00:00:02,000"
        clip_srt += f"1\n{start_time_str} --> {end_time_str}\n[封面]\n\n"

    print(f"总片段: {total_clips}")    
    for i, (start, end, text, kw) in enumerate(total_clips):
        start = max(0, start)
        end = min(video.duration, end)
        if end <= start:
            continue
        clip_vid = video.subclip(start, end)
        video_clips.append(clip_vid)
        
        # 字幕索引需要考虑封面（如果有）
        subtitle_index = i + (2 if cover_clip is not None else 1)
        subtitle_items.append(((current_time, current_time + (end - start)), kw))
        
        # 创建SRT条目
        hours_start = int(current_time // 3600)
        minutes_start = int((current_time % 3600) // 60)
        seconds_start = int(current_time % 60)
        milliseconds_start = int((current_time % 1) * 1000)
        
        hours_end = int((current_time + (end - start)) // 3600)
        minutes_end = int(((current_time + (end - start)) % 3600) // 60)
        seconds_end = int((current_time + (end - start)) % 60)
        milliseconds_end = int(((current_time + (end - start)) % 1) * 1000)
        
        start_time_str = f"{hours_start:02d}:{minutes_start:02d}:{seconds_start:02d},{milliseconds_start:03d}"
        end_time_str = f"{hours_end:02d}:{minutes_end:02d}:{seconds_end:02d},{milliseconds_end:03d}"
        
        clip_srt += f"{subtitle_index}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
        
        current_time += (end - start)

    if not video_clips:
        raise ValueError("无可剪辑片段")

    # 如果提供了中间插入的图片，则处理这些图片
    if image_inserts:
        video_clips, subtitle_items, clip_srt, current_time = insert_images_into_video(
            video_clips, subtitle_items, clip_srt, image_inserts, video.size, video.fps
        )

    # 8. 合并视频
    final_video = concatenate_videoclips(video_clips) if len(video_clips) > 1 else video_clips[0]

    # 9. 添加字幕
    def make_textclip(txt):
        return TextClip(
            txt,
            font=font_path,
            fontsize=font_size,
            color='white',
            stroke_color='white',
            stroke_width=2,
            size=final_video.size,
            method='caption',
            align='south'
        )
    
    if subtitle_items:
        subtitles = SubtitlesClip(subtitle_items, make_textclip)
        final_video = CompositeVideoClip([final_video, subtitles.set_pos(('center', 'bottom'))])

    # 10. 叠加 BGM
    if bgm_path and os.path.exists(bgm_path):
        bgm = AudioFileClip(bgm_path)
        if bgm.duration < final_video.duration:
            bgm = afx.audio_loop(bgm, duration=final_video.duration)
        else:
            bgm = bgm.subclip(0, final_video.duration)
        bgm = bgm.fx(afx.audio_fadein, 1.0).fx(afx.audio_fadeout, 1.0)
        
        original_audio = final_video.audio
        if original_audio:
            final_audio = original_audio.volumex(0.6).fx(afx.composite, bgm.volumex(0.4))
        else:
            final_audio = bgm.volumex(0.5)
        final_video = final_video.set_audio(final_audio)

    # 11. 导出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=output_path.replace('.mp4', '_temp.m4a'),
        remove_temp=True,
        logger='bar',
        threads=min(8, os.cpu_count() or 4), # 启用多线程写入
        ffmpeg_params=[
            '-preset', 'ultrafast',     # ⚡ 最快编码速度（牺牲压缩率）
            '-tune', 'fastdecode',      # 优化解码速度（可选）
            '-threads', '0',            # 自动使用所有 CPU 核心（等价于 -threads auto）
            '-x264-params', 'nal-hrd=cbr',  # 避免变码率波动（可选）
        ]
    )
    logging.info(f"✅ 推广视频已生成: {output_path}")
    
    # 返回与现有接口兼容的输出格式
    message = f"成功生成推广视频，包含 {len(total_clips)} 个片段"
    return output_path, None, message, clip_srt

# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="AI 短剧推广视频自动剪辑（LLM 动态生成高能关键词）")
    parser.add_argument("--video",default="D:\\pythonwork\\FunClip\\data\\merged_video_ae02936d.mp4", help="原始视频路径")
    parser.add_argument("--srt", default="D:\\pythonwork\\FunClip\\data\\srt.srt", help="SRT 字幕文件路径")
    parser.add_argument("--output", default="D:\\pythonwork\\FunClip\\data\\output\\merged_video_ae02936d.mp4", help="输出视频路径")
    parser.add_argument("--llm", default="qwen3-max", help="LLM 模型名（如 qwen, llama3, gpt-4o）")
    parser.add_argument("--llm_url", default=None, help="LLM API 地址（如 http://localhost:11434）")
    parser.add_argument("--bgm", default=None, help="背景音乐路径（可选）")
    parser.add_argument("--font", default="./font/STHeitiMedium.ttc", help="中文字体路径")
    parser.add_argument("--max_clips", type=int, default=5, help="最大片段数量（含钩子）")
    parser.add_argument("--expand", type=float, default=1.0, help="片段前后扩展秒数")

    args = parser.parse_args()
   
    create_promo_video(
        video_path=args.video,
        srt_content=args.srt,
        output_path=args.output,
        llm_model=args.llm,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        bgm_path=args.bgm,
        font_path=args.font,
        expand_sec=args.expand,
        max_clips=args.max_clips
    )

if __name__ == "__main__":
    main()
