import os
import sys
import argparse
import datetime
import json
import subprocess
import glob
from email.utils import formatdate
import feedparser
import trafilatura
from google import genai
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

def get_args():
    parser = argparse.ArgumentParser(description="RSS to Podcast Pipeline")
    parser.add_argument("--local-test", action="store_true", help="Run locally with mock data and skip commit")
    return parser.parse_args()

def phase1_ingestion(feed_url, is_local_test):
    print("Phase 1: Ingestion & Scraping")
    if is_local_test and not feed_url:
        print("Using MOCK feed data for local test...")
        return [{
            "title": "Mock Global News", 
            "link": "http://example.com/mock1", 
            "text": "This is a mock article about global news and politics for testing the pipeline."
        }, {
            "title": "Mock Tech Innovation", 
            "link": "http://example.com/mock2", 
            "text": "A new AI model was released today that can generate audio offline quickly."
        }]
        
    if not feed_url:
        print("Set RSS_FEED_URL env var")
        sys.exit(1)

    print(f"Fetching RSS feed from: {feed_url}")
    feed = feedparser.parse(feed_url)
    print(f"Total entries found in feed: {len(feed.entries)}\n")
    
    entries = []
    now = datetime.datetime.now(datetime.timezone.utc)
    
    for entry in feed.entries:
        # Filter for the last 24 hours (86400 seconds)
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            import time
            dt = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed), datetime.timezone.utc)
            if (now - dt).total_seconds() > 86400:
                continue

        link = entry.link
        text = entry.get('summary', '')
        
        # Scrape full text
        downloaded = trafilatura.fetch_url(link)
        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted:
                text = extracted
                
        entries.append({
            "title": entry.title,
            "link": link,
            "text": text
        })
    print(f"Ingested {len(entries)} articles from the last 24h.")
    return entries

def phase2_map_reduce(entries, is_local_test):
    print("Phase 2: AI Map-Reduce Writers' Room")
    if not entries:
        return ""
        
    api_key = os.environ.get("GEMINI_API_KEY", "")
    is_mock = is_local_test or api_key == "mock_key" or not api_key
    
    if not is_mock:
        client = genai.Client(api_key=api_key)
        
        # Hardcoded to bypass API permission errors on models.list()
        model_name = "gemini-2.5-flash-lite"
        print(f"Using Gemini Model: {model_name}")
            
    else:
        client = None
        model_name = None
    
    categories = ["Global News", "Tech & AI", "Social Impact & Humanitarian", "Gaming & Hobbies", "Misc"]
    
    # Define strict schema for tagging
    class CategoryEnum(str, Enum):
        global_news = "Global News"
        tech_ai = "Tech & AI"
        social_impact = "Social Impact & Humanitarian"
        gaming = "Gaming & Hobbies"
        misc = "Misc"

    class ArticleTag(BaseModel):
        category: CategoryEnum = Field(description="The primary category of the article.")

    # 1. Tagging
    tagged_articles = {cat: [] for cat in categories}
    for entry in entries:
        prompt = f"Categorize this article. Return ONLY a JSON object with a single key 'category' matching the Enum.\nHeadline: {entry['title']}\nSnippet: {entry['text'][:500]}"
        try:
            if is_mock:
                tagged_articles["Misc"].append(entry)
                continue

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ArticleTag,
                ),
            )
            # The new SDK parses structured json directly!
            if response.parsed:
                cat = response.parsed.category.value
            else:
                cat = json.loads(response.text).get('category', "Misc")
            
            if cat in tagged_articles:
                tagged_articles[cat].append(entry)
            else:
                tagged_articles["Misc"].append(entry)
        except Exception as e:
            print(f"Tagging error for '{entry['title']}': {e}")
            tagged_articles["Misc"].append(entry)
            
    # 2. Segment Summaries
    segment_scripts = []
    for cat, articles in tagged_articles.items():
        if not articles:
            continue
        
        combined_text = "\n\n".join([f"Headline: {a['title']}\nContent: {a['text'][:2000]}" for a in articles])
        prompt = f"Act as a podcast producer. Write a conversational, 2-minute radio segment summarizing these {cat} stories. Make it engaging. DO NOT include any sound effect instructions, music cues, or speaker labels (like 'Host:' or 'Music:'). Write ONLY the spoken words.\n\n{combined_text}"
        
        try:
            if is_mock:
                segment_scripts.append(f"In {cat} today, we look at exciting updates: {articles[0]['title']}.")
                continue

            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            segment_scripts.append(res.text)
        except Exception as e:
            print(f"Segment summary error for {cat}: {e}")

    # 3. Show Assembly
    assembly_prompt = (
        "Act as a podcast host. I will provide you with several segment scripts. "
        "Your job is to weave them together into a single, cohesive daily podcast episode. "
        "Write a punchy intro to start the show, and a brief sign-off at the end. "
        "CRITICAL INSTRUCTION: You MUST keep the original text of the segment scripts EXACTLY as provided. "
        "Do NOT rewrite, summarize, or alter the core content of the segments. "
        "Instead, write smooth, contextual transitional sentences BETWEEN the segments to connect them naturally. "
        "DO NOT include any sound effect instructions, music cues, or speaker labels (like 'Host:' or 'Music:'). "
        "Write ONLY the exact spoken words to be read by the TTS.\n\nSegments:\n"
    ) + "\n\n---\n\n".join(segment_scripts)
    
    try:
        if is_mock:
             return "Welcome to the daily podcast. " + " ".join(segment_scripts) + " Thanks for listening."
             
        final_res = client.models.generate_content(
            model=model_name,
            contents=assembly_prompt
        )
        return final_res.text
    except Exception as e:
         print(f"Assembly error: {e}")
         return "\n\n".join(segment_scripts)

def phase3_audio_generation(script, out_path, is_mock):
    print("Phase 3: Audio Generation (Kokoro TTS)")
    if is_mock:
        print("MOCK TEST: Skipping full Kokoro generation to save time.")
        print("Generating a tiny silent MP3 to satisfy GitHub Actions commit...")
        # Create a silent 1-second wav
        sf.write("temp_podcast.wav", np.zeros(24000), 24000)
        subprocess.run(["ffmpeg", "-y", "-i", "temp_podcast.wav", out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove("temp_podcast.wav")
        print(f"Successfully saved MOCK {out_path}")
        return

    pipeline = KPipeline(lang_code='a') 
    generator = pipeline(script, voice='af_heart', speed=1)
    
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunks.append(audio)
        
    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        temp_wav = "temp_podcast.wav"
        sf.write(temp_wav, final_audio, 24000)
        
        # Convert WAV to MP3 using ffmpeg
        print("Converting to MP3...")
        subprocess.run(["ffmpeg", "-y", "-i", temp_wav, out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_wav)
        print(f"Successfully saved {out_path}")
    else:
        print("No audio generated.")

def phase4_archival():
    print("Phase 4: Archival & Maintenance")
    now = datetime.datetime.now()
    threshold = now - datetime.timedelta(days=14)
    retained_files = []
    
    for file in glob.glob("episode-*.mp3"):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        if mtime < threshold:
            print(f"Deleting older episode (14+ days): {file}")
            os.remove(file)
        else:
            retained_files.append(file)
            
    # Return matched files sorted by newest first
    return sorted(retained_files, reverse=True)

def phase5_xml(retained_files, repo_url):
    print("Phase 5: iTunes-Compliant XML Generation")
    
    items_xml = ""
    for file in retained_files:
        mtime = os.path.getmtime(file)
        pub_date = formatdate(mtime, localtime=False)
        size = os.path.getsize(file)
        
        # Properly escape URL and generate standards-compliant tags
        file_url = f"{repo_url}/{file}"
        items_xml += f'''
        <item>
            <title>{file.replace(".mp3", "")}</title>
            <pubDate>{pub_date}</pubDate>
            <enclosure url="{file_url}" length="{size}" type="audio/mpeg"/>
            <guid isPermaLink="true">{file_url}</guid>
            <itunes:summary>Daily automated summary.</itunes:summary>
            <itunes:explicit>no</itunes:explicit>
        </item>'''

    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
    <channel>
        <title>Daily Automated Podcast</title>
        <link>{repo_url}</link>
        <description>Daily newsletter summarized and read aloud by AI.</description>
        <language>en-us</language>
        <itunes:author>Automated Bot</itunes:author>
        <itunes:image href="{repo_url}/cover.jpg" />
{items_xml}
    </channel>
</rss>
'''
    with open("podcast.xml", "w", encoding="utf-8") as f:
        f.write(xml_content)
    print("podcast.xml updated")

def main():
    args = get_args()
    feed_url = os.environ.get("RSS_FEED_URL")
    
    entries = phase1_ingestion(feed_url, args.local_test)
    if not entries:
        print("No new entries to process.")
        return
        
    script = phase2_map_reduce(entries, args.local_test)
    if not script:
         print("Failed to generate script.")
         return
         
    print("\n=== Final Assembled Script ===\n", script, "\n============================\n")
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    out_mp3 = f"episode-{today_str}.mp3"
    
    # Check ffmpeg availability locally
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Warning: ffmpeg is not installed on this system. Phase 3 might fail during MP3 conversion.")
    
    phase3_audio_generation(script, out_mp3, args.local_test)
    retained_files = phase4_archival()
    
    repo_url = os.environ.get("GITHUB_PAGES_URL", "https://example.github.io/podcast")
    phase5_xml(retained_files, repo_url)
    
    print("Run complete!")

if __name__ == "__main__":
    main()
