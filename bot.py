import os
from datetime import datetime
import discord
from piemanese.translator import Translator

def main():
    assert 'DISCORD_USER_IDS' in os.environ
    assert 'DISCORD_TOKEN' in os.environ

    translator = Translator()
    client = discord.Client()
    user_ids = os.environ['DISCORD_USER_IDS'].split(',')

    @client.event
    async def on_ready():
        print('Logged in as', client.user)

    @client.event
    async def on_message(msg):
        if not msg.content:
            return
        if msg.author == client.user:
            return
        if msg.guild and msg.author.id not in user_ids:
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{now}] [{msg.channel}] {msg.author}: {msg.content}')
        msg_clean = ' '.join(translator.tokenize(msg.content))
        msg_translated = translator(msg_clean)
        if msg_translated and msg_clean != msg_translated:
            await msg.channel.send(msg_translated)
            print('translation:', msg_translated)

    client.run(os.environ['DISCORD_TOKEN'])

if __name__ == '__main__':
    main()
