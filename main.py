# coding=utf-8

import pandas as pd
import numpy as np

from ffnet import ffnet, mlgraph, savenet

def run_network():
    # Generate standard layered network architecture and create network
    conec = mlgraph((14,28,1))
    net = ffnet(conec)

    df = pd.read_csv('data/copacabana.csv', sep=';')
    variables = [
        'Posicao', 'Quartos', 'Vagas', 'DistIpanema', 'DistPraia',
        'DistFavela', 'RendaMedia', 'RendaMovel', 'RendaMovelRua',
        'Vu2009', 'Mes', 'Idade', 'Tipologia', 'AreaConstruida'
    ]
    input = df[variables]
    target = df[['VAL_UNIT']]

    # Train network
    #first find good starting point with genetic algorithm (not necessary, but may be helpful)
    print "FINDING STARTING WEIGHTS WITH GENETIC ALGORITHM..."
    net.train_genetic(input, target, individuals=20, generations=500)
    #then train with scipy tnc optimizer
    print "TRAINING NETWORK..."
    net.train_tnc(input, target, maxfun = 1000, messages=1)

    print "TESTING NETWORK..."
    output, regression = net.test(input, target, iprint=0)

    # Save/load/export network
    print "Network is saved..."
    savenet(net, "data/capacabana.net")

    return output, regression
