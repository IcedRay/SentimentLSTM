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
rnn = RNN(87, 400, 2)
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
hidden =torch.zeros(1, 400)
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
Résultat : """
tensor([[-0.6666, -0.7205]], grad_fn=<LogSoftmaxBackward>)
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
torch.Size([5, 1, 87])
category = 0 / line = Au travail, pas content. Faites-en de ce qui en à de meilleur
category = 0 / line = Je pense que je deviens malade
category = 1 / line = Haha, je reçois les os d'une politique de non-diva - accord réciproque avec les clubs de phoques, manteau de fourrure gratuit pour chaque diva. J'aime!
category = 1 / line = impressionnant! J'ai été un utilisateur frileux depuis le 1er jour de la mac version bêta, donc c'est génial de voir à quel point il est venu (et continuera!)
category = 1 / line = Fêtard dans hollyweird, cali. Uhhh yeahhhh !!!!! Luv my hotties !!!
category = 0 / line = Qui est l'autre gars? Pourquoi les dames célibataires ont-elles toujours utilisé les paris après les tentatives de joes, je me sens un peu honteux de le faire: /
category = 0 / line = Est déprimé après avoir visionné des documentaires sur la pêche illicite des requins
category = 0 / line = As-tu regardé le français ouvert? très excitant! Nadal a perdu aujourd'hui
category = 0 / line = Lundi + premier du mois = longggg dayyyyy.
category = 1 / line = Lol, ce sera un rire ... on peut tous s'asseoir et se joindre à l'autre comme nous buvons !! hahaha
0 0% (0m 0s) 0.6856 Je ne sais pas si je l'avais imaginé. Et je ne sais pas si je lui manque. / 0 ✓
5000 5% (1m 46s) 0.5112 : Aussi, n'importe qui peut acheter un domaine .co.uk, donc ce n'est pas aussi clair que possible. / 0 ✓
10000 10% (3m 32s) 0.9647 Je pourrais peut-être laisser ces billets d'idole américaines partir parce que personne ne se rendra à Rochester avec moi pour les ramasser! Soo Maddddddd !!! / 1 ✗ (0)
15000 15% (5m 18s) 1.0711 Alors je serais incroyable maintenant! / 1 ✗ (0)
20000 20% (7m 3s) 0.6617 Ughh maths aujourd'hui ..... oh bien mal, de toute façon, le clavier et la science, 2morrow, l'histoire et les harcis espagnols. Et anglais jeudi / 0 ✓
25000 25% (8m 48s) 0.7765 Aww that is no good / 1 ✗ (0)
30000 30% (10m 35s) 0.6252 Je viens de montrer à maman ces images comme Woww c'est tellement sympa. Amusez-vous, obtenez du bronzage. Mais pas tanneur que moi / 1 ✓
35000 35% (12m 21s) 0.4224 Je pensais que j'avais une configuration pour obtenir du thon, mais je me suis trompé. / 0 ✓
40000 40% (14m 6s) 0.6975 J'ai eu la photo, pas un seul de moi / 1 ✗ (0)
45000 45% (15m 52s) 0.7967 Yip. L'odeur du fromage rôti s'arrête / 0 ✗ (1)
50000 50% (17m 39s) 0.5871 Au-dessus de ça, je n'ai toujours pas vu Hangova, alors, qui veut me prendre ou moins me faire compagnie ... plz / 0 ✓
55000 55% (19m 25s) 0.9454 A peur de l'éclairage et du tonnerre !!!! / 1 ✗ (0)
60000 60% (21m 12s) 0.7483 Escalier pour anime heaven / 0 ✗ (1)
65000 65% (22m 56s) 0.5874 Salut ma sœur presque lol shea m'a dit qu'il t'avait vu la nuit dernière, je serais allé mais mes soeurs ne sont pas là 2 me préparent / 0 ✓
70000 70% (24m 42s) 0.6495 Oh, bonjour, j'espère publier des notes sur le blog des jeux le lendemain, si vous êtes intéressé! / 0 ✓
75000 75% (26m 28s) 0.6880 c'est! Rb: "c'est gentil rb" ♫ ~ 7kb9f / 1 ✓
80000 80% (28m 12s) 0.7090 Temps pour de la nourriture gratuite / 0 ✗ (1)
85000 85% (29m 57s) 0.6350 J'ai une bouée de sauvetage à portée de main. / 1 ✓
90000 90% (31m 44s) 0.3978 Les citrons en colère @ the stardust club ce soir! 7643 firestone blvd., Downey, californie 90241 viennent soutenir! Je vais vous donner un citron! / 1 ✓
95000 95% (33m 30s) 0.7858 Bonbons / bonbons ... sucre ... c'est tout ce qu'il faut pour s'installer ... puis-je ajouter de la caféine / 0 ✗ (1)
100000 100% (35m 25s) 0.6897 Est à l'intérieur hors de la pluie / 0 ✓
"""
