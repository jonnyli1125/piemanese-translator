import argparse
import json
import os
import re
import praw
import markdown
from bs4 import BeautifulSoup

SPACE_RE = re.compile(r'\s+')

def clean(line):
    line = markdown.markdown(line)
    line = BeautifulSoup(line, 'html.parser').get_text()
    line = line.replace('[deleted]', '').replace('[removed]', '')
    line = SPACE_RE.sub(' ', line).strip()
    return line

def write_to_file(subreddit, lines):
    clean_lines = []
    for line in lines:
        clean_line = clean(line)
        if clean_line:
            clean_lines.append(clean_line)
    out_path = os.path.join(args.output_dir, f'{subreddit}.txt')
    with open(out_path, 'a', encoding='utf-8') as f:
        f.writelines(f'{line}\n' for line in clean_lines)

def main(args):
    with open('reddit_crawler_config.json', 'r') as f:
        config = json.load(f)
    reddit = praw.Reddit(**config)

    os.makedirs(args.output_dir, exist_ok=True)

    for subreddit_name in args.subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=args.limit):
            lines = [submission.title]
            if submission.selftext:
                lines.append(submission.selftext)
            while True:
                try:
                    submission.comments.replace_more(limit=None)
                    break
                except e:
                    print(e)
            comments = submission.comments.list()
            for comment in comments:
                lines.append(comment.body_html)
                if len(lines) >= 1000:
                    write_to_file(subreddit_name, lines)
                    lines.clear()
            if lines:
                write_to_file(subreddit_name, lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('subreddits', nargs='+')
    parser.add_argument('--limit', default=1000)
    args = parser.parse_args()

    main(args)
