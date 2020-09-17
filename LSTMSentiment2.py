#----------------------------------
# Rayane segueg 2020
#----------------------------------
# Reseau de neurone sans poids
from time import *
from random import *
from turtle import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#dataframe : taille 1526723
# 80% = 1221378 (les 20% restant sont reservés aux tests)
n_letters = 87
dataframe = pd.read_csv('french_tweets.csv', sep=',',header=0)
#print(dataframe.values)
def charToArray(char):
    k = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZâàêéèîïoöùûç-'\".,;:!1234567890%€$& " #87array
    output = list()
    for i in k:
        output.append(int(i==char))
    return output

def letterToIndex(char):
    k = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZâàêéèîïoöùûç-'\".,;:!1234567890%€$& " #87array
    for i in range(len(k)):
        if k[i]==char:
            return i
    return 86
    
# Je le tranforme en array de bool

##############################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(87, 30, 2)
##############################
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i , category_i

def letterToTensor(letter):
    tensor = torch.zeros(1, 87)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

inpt = letterToTensor('A')
hidden =torch.zeros(1, 30)
output, next_hidden = rnn(inpt, hidden)
print(output)

def randomChoice():
    return randint(int(0.1*1526723), int(0.9*1526723))

def randomTrainingExample():
    rd = randomChoice()
    line = dataframe.at[rd,"text"]
    category = dataframe.at[rd,"label"]
    category_tensor = torch.tensor([category], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
#######################################
criterion = nn.NLLLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

#######################################
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time()

for iter in range(0, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# oui j'ai sauvagement copié et modifié le tutoriel, mais ça réponds exactement a la problématique
résultat = """
tensor([[-0.8100, -0.5886]], grad_fn=<LogSoftmaxBackward>)
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
torch.Size([5, 1, 87])
category = 0 / line = je connais!! C'est vous qui l'avez recommandé! Ressemble à ce dernier lot n'aura que 2 livres. Mieux que 0!
category = 1 / line = Haha assis dans la voiture. La première fois que j'utilise Twitter sur mon téléphone
category = 0 / line = En route vers les séances d'entraînement
category = 0 / line = Je vais me détendre pour la nuit !!! Pourrait courir, mais il semble qu'il puisse pleuvoir
category = 0 / line = Travailler jusqu'à 230 se sentir comme une merde
category = 0 / line = Les vents forts et les pluviométries au hasard provoquent une promenade imposable
category = 1 / line = Merci de vous amuser au travail lol xxx
category = 0 / line = Non, il est encore loin
category = 0 / line = Arrêtez de me rendre jaloux avec tous ces twitpics que vous continuez à poster! Je veux une photo avec toi
category = 0 / line = Je n'ai pas eu assez de soleil 2 jours!
0 0% (0m 0s) 0.7564 Les devoirs principalement. toi? / 1 ✗ (0)
5000 5% (0m 42s) 0.6705 je t'aime. c'est tout / 1 ✓
10000 10% (1m 25s) 0.6339 Est à l'intérieur hors de la pluie / 0 ✓
15000 15% (2m 8s) 0.7704 De retour à l'église, je vais. La conférence continue / 0 ✗ (1)
20000 20% (2m 51s) 0.6934 Merci, je viens d'apprendre que je ne peux pas avoir Skype sur mon mobile boo / 1 ✗ (0)
25000 25% (3m 34s) 0.6729 je ne sais pas. Je vais tester mon gps thinger sur le bb. Tenir les pouces / 1 ✓
30000 30% (4m 15s) 0.6751 Haha okay Je ne peux pas attendre pour voir mon ambre, elle m'a envoyé un message textuel à 10 heures du matin ce matin, disant «réveillez-vous ... chienne. Je t'aime 'xd lol / 1 ✓
35000 35% (4m 57s) 0.6620 Je n'ai pas dormi la nuit dernière parce que disney / 1 ✓
40000 40% (5m 39s) 0.8045 Regarder les feux d'artifice partir à Boston depuis la fenêtre de ma chambre. impressionnant / 0 ✗ (1)
45000 45% (6m 22s) 0.9050 J'aime le son de votre cookie. C'est ... ehrm ... croquant ... lol. En attendant avec impatience de vous voir le 2 août. / 0 ✗ (1)
50000 50% (7m 4s) 0.6128 Je l'apprécie beaucoup merci beaucoup / 1 ✓
55000 55% (7m 46s) 0.6379 Au travail, je vais / 0 ✓
60000 60% (8m 28s) 0.6986 Non wi fi au check in hun / 1 ✗ (0)
65000 65% (9m 10s) 0.8688 triste jour. En pensant à ma gramme ... j'aimerais être là avec eux / 1 ✗ (0)
70000 70% (9m 51s) 0.5782 Je suis actuellement sur mon troisième jour de recherche de mon premier livre harry potter. J'ai honte de le perdre. / 0 ✓
75000 75% (10m 33s) 0.3677 C'est la bonne façon de nettoyer aujourd'hui! / 1 ✓
80000 80% (11m 15s) 0.6939 Je suis finalement allongé près de la piscine! La peau pâle doit aller .... je sais que je suis un problème pour demander le homard rouge d'abord ... weaksauce lol / 1 ✗ (0)
85000 85% (11m 56s) 1.0114 Oh regarde! C'est moi en vérité / 0 ✗ (1)
90000 90% (12m 38s) 0.7572 Suis-je le seul à faire? / 1 ✗ (0)
95000 95% (13m 21s) 0.8522 Une nuit de 3/4 ce soir - semble merveilleux contre un ciel bleu clair / 0 ✗ (1)
100000 100% (14m 2s) 0.8285 Hmm .. une vente sur les pièces de rechange pour mon projecteur. Pour le prix d'un réservoir de gaz, je pense que je devrais y regarder. / 0 ✗ (1)
"""
torch.save(rnn.state_dict(),"Sentimentlstm.nn" )
