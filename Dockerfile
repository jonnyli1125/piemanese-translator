FROM python:3.10
WORKDIR /code
COPY . /code
RUN apt-get update && apt-get install -y espeak-ng
RUN pip install discord.py dill nltk phonemizer panphon
CMD python -u bot.py