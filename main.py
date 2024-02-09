import discord
from discord.ext import commands
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='#', intents=intents)#префикс для команды бота

@bot.event
async def on_ready():#при команде #on_ready будет выходить сообщения
    print(f'We have logged in as {bot.user}')#само сообщения

def detect_car(image, model, label):
    np.set_printoptions(suppress=True)
    model = load_model(model, compile=False)#Эта строка загружает модель машинного обучения из файла
    class_names = [line.strip() for line in open(label, "r").readlines()]
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 0
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    #список машин
    audi = 'ауди'
    bmw = 'бмв'
    bugatti = 'бугати'
    ferrari = 'ферари'
    jaguar = 'ягуар'
    Lamborghini = 'ламборгини'
    mclaren = 'макларен'
    mersedes = 'мерседес'
    pagani = 'пигини'
    tesla = 'тесла'
    car= ''
    if class_name[0] == '0':
        car = audi
    elif class_name[0] == '1':
        car = bmw
    elif class_name[0] == '2':
        car = bugatti
    elif class_name[0] == '3':
        car = ferrari
    elif class_name[0] == '4':
        car = jaguar
    elif class_name[0] == '5':
        car = Lamborghini
    elif class_name[0] == '6':
        car = mclaren
    elif class_name[0] == '7':
        car = mersedes
    elif class_name[0] == '8':
        car = pagani
    elif class_name[0] == '9':
        car = tesla
    return class_name[2:], confidence_score, car

#cars: 
#0 AUDI
#1 bmw
#2 bugatti
#3 ferrari
#4 jaguar
#5 Lamborghini
#6 mclaren
#7 mersedes
#8 pagani
#9 tesla

@bot.command()
async def photo(ctx):
    if ctx.message.attachments:
        for imga in ctx.message.attachments:
            p = imga.filename
            o = imga.url
            await imga.save(f'./{imga.filename}')
            result = detect_car(image=f'./{imga.filename}', model="keras_model.h5", label="labels.txt")
            await ctx.send(result)

bot.run("YOUR-TOKEN")
