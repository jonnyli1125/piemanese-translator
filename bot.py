import json
from datetime import datetime
import discord
from translator import Translator

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    translator = Translator()
    client = discord.Client()

    @client.event
    async def on_ready():
        print('Logged in as', client.user)

    @client.event
    async def on_message(msg):
        if not msg.content:
            return
        if msg.author == client.user:
            return
        if msg.guild and msg.author.id not in config['user_ids']:
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{now}] [{msg.channel}] {msg.author}: {msg.content}')
        msg_clean = ' '.join(msg.content.lower().split())
        msg_translated = translator(msg_clean)
        if msg_translated != msg_clean:
            await msg.channel.send(msg_translated)
            print('translate:', msg_translated)

    client.run(config['token'])

if __name__ == '__main__':
    main()
