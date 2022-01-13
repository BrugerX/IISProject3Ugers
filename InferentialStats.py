import random
import math
import numpy as np
from sklearn.preprocessing import minmax_scale

#Class for samples
class sampleDataSet:
    def __init__(self,initSample,initIsPopulation = False,initSampleSize = 0, initIsProportion = False,initAlpha = 0):
        #Settings
        self.alpha = initAlpha #For calculating the C.I %
        self.isPopulation = initIsPopulation #The population from which samples are taken
        self.sampleSize = initSampleSize #Sample size for random sample generation
        self.isProportion = initIsProportion #Whether or not we're dealing with proportions

        #T-Test
        self.tScores = [6.314,2.92,2.353,2.132,2.015,1.943,1.895,1.86,1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,1.74,1.734,1.729,1.725,1.721,1.717,1.714,1.711,1.708,1.706,1.703,1.701,.1699,1.697]

        #Sample data
        self.sample = initSample
        self.populationSize = len(initSample)
        if self.isPopulation != False and self.sampleSize != 0: #Calculates random sample
            self.simpleRandomSample()
        self.n = len(self.sample)

        #Central tendency
        self.sampEstMean = 0
        self.calcMean()

        #Proportion
        if self.isProportion == True:
            self.pino = (self.sampEstMean*self.n+2)/(self.n+4)  # "P med bølge over" til Agresti Coull
            self.nino = (self.n +4)


        #Samp Dispersion
        self.sampStandDev = 0
        self.popStandDev = 0
        self.calcStandDev()
        self.sampVar = self.sampStandDev**2

        #Inferential stats
        self.standErrMean = 0
        self.calcStandErrMean()
        self.confI = 0
        self.calc95ConfI()

        #Sample size estimation
        #TODO: Lav en method, der fortæller, hvor stort et sample size, vi skal bruge

    #Laver en liste med n antal tilfældige elementer fra populationen
    def simpleRandomSample(self):
        randomSample = []
        for x in range(self.sampleSize):
            randomSample = randomSample + [self.sample[random.randrange(0, len(self.sample))]]
        self.sample = randomSample

    #Beregner mean
    def calcMean(self):
        self.sampEstMean = sum(self.sample)/self.n


    #Beregner variance på baggrund af et sample
    def calcStandDev(self):
        if self.isProportion == True:
            self.sampStandDev = math.sqrt(self.sampEstMean*(1-self.sampEstMean)) #sqrt(p*(1-p))
        else:
            xminm =  sum((np.array(self.sample) - self.sampEstMean)**2)


            self.sampStandDev = math.sqrt((1/(self.n-1)) * xminm) #SampleStandDev
            self.popStandDev = math.sqrt((1/(self.n)) * xminm) #Population stand dev

    def calcStandErrMean(self):
        self.standErrMean = self.sampStandDev/math.sqrt(self.n)

    def calc95ConfI(self):


        if self.alpha != 0: #Hvis det ikke er 1.96 gange standard deviation
            ratio95 = self.alpha
        elif self.n >= 30: #Hvis det bare er normalt 95%-coverage C.I
            ratio95 = 1.96
        else : #Laver t score, hvis n <30
            ratio95 = self.tScores[self.n-2]

        self.confI = ratio95*self.standErrMean


        if (0.9<self.confI or self.confI<0.1) and self.isProportion == True: #Agresti-Coull
            self.standErrMean = math.sqrt((self.pino *(1-self.pino)/self.nino))
            self.confI = ratio95 * self.standErrMean

        self.cInterval = [self.sampEstMean - self.confI, self.sampEstMean + self.confI]



    #Beregner variance på baggrund af en population
    def popVariance(sample):
        sampleMean = calcMean(sample)
        diffSampleMeanxSquared =  [(x - sampleMean)**2 for x in sample]
        sumDiffSampleMeanxSquared = sum(diffSampleMeanxSquared)
        diffi = 1/len(sample)
        sampleVariance = (math.sqrt(diffi*sumDiffSampleMeanxSquared))**2
        return sampleVariance


    #Beregner variance vha. Populationens variance
    def sampleVarianceFromPop(population,sample):
        sampleVarFrompop = popVariance(population)/len(sample)
        return sampleVarFrompop


""" It assumes the interval to be 95%"""
def sampleSizeCalc(p,e):
    k = 1.96**2
    top = p*(1-p)
    try:
        n = k*top/e**2
        return(n)
    except:
        print(f"ERROR in calculate sample size\nThis is e: {e}")




