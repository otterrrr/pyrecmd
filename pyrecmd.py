import numpy as np
import math
import random

class PyRecmd:

    ITEM = 0
    USER = 1
    
    def __init__(self, num_components, ratings, minrate=None, maxrate=None, itemnames=None, usernames=None):
        rates = [rel[2] for rel in ratings]
        self.n = num_components
        self.minrate = minrate if minrate else min(rates)
        self.maxrate = maxrate if maxrate else max(rates)
        self.rateconv = lambda x: (min(max(x,self.minrate),self.maxrate)-self.minrate)/(self.maxrate-self.minrate)
        self.rateinvconv = lambda x: min(max(x,0),1)*(self.maxrate-self.minrate)+self.minrate
        self.seqmap = [
            { item:i for i, item in enumerate(sorted(frozenset([rel[0] for rel in ratings])))}, # item-to-seq map
            { user:i for i, user in enumerate(sorted(frozenset([rel[1] for rel in ratings])))}  # user-to-seq map
        ]
        self.srcmap = [
            [ item for item, seq in sorted(self.seqmap[self.ITEM].items(),key=lambda x:x[1])], # seq-to-item map
            [ user for user, seq in sorted(self.seqmap[self.USER].items(),key=lambda x:x[1])]  # seq-to-user map
        ]
        self.namemap = [
            [itemnames[item] for item in self.srcmap[self.ITEM]] if itemnames else self.srcmap[self.ITEM],
            [usernames[user] for user in self.srcmap[self.USER]] if usernames else self.srcmap[self.USER]
        ]
        self.shape = [
            len(self.srcmap[self.ITEM]),
            len(self.srcmap[self.USER])
        ]
        self._initialize_weight()
        self.src = [ (self.seqmap[self.ITEM][rel[0]], self.seqmap[self.USER][rel[1]], rel[2]) for rel in ratings ]
        self.Y = np.zeros(self.shape)
        self.R = np.zeros(self.shape,dtype=int)
        for rel in self.src:
            self.Y[rel[0],rel[1]] = self.rateconv(rel[2])
            self.R[rel[0],rel[1]] = 1
        self.weakside = self.USER if np.mean(np.sum(self.R, axis=0)) < np.mean(np.sum(self.R,axis=1)) else self.ITEM
        self.cached = {}
    
    def _initialize_weight(self):
        # weight matrix for both item and user
        self.W= [
            np.fabs(np.random.normal(0,1/math.sqrt(self.n),(self.shape[self.ITEM],self.n))),
            np.fabs(np.random.normal(0,1/math.sqrt(self.n),(self.shape[self.USER],self.n)))
        ]
        # first-order-moment matrix for both item and user weight
        self.M1 = [
            np.zeros(self.W[self.ITEM].shape),
            np.zeros(self.W[self.USER].shape)
        ]
        # second-order-moment matrix for both item and user weight
        self.M2 = [
            np.zeros(self.W[self.ITEM].shape),
            np.zeros(self.W[self.USER].shape)            
        ]
    
    def _apply_validation_split(self, validationsplit):
        Rtrain = np.zeros(self.Y.shape,dtype=int)
        Rtest = np.zeros(self.Y.shape,dtype=int)
        wa = self.weakside
        sa = 0 if wa else 1
        axis_item_map = {}
        for rel in self.src:
            if rel[wa] not in axis_item_map:
                axis_item_map[rel[wa]] = []
            axis_item_map[rel[wa]].append(rel[sa])
        for k,vals in axis_item_map.items():
            random.shuffle(vals)
            for i,v in enumerate(vals):
                (Rtrain if (i+1)/len(vals) > validationsplit else Rtest)[v if wa else k,k if wa else v] = 1
        return (Rtrain, Rtest)
    
    def fit(self, learningrate=3e-4, regularization=1.0, iterations=1e5, validationsplit=0.0, initializeweight=True, beta1=0.9, beta2=0.999, verbose=False):
        
        if initializeweight:
            self._initialize_weight()
        
        if validationsplit not in self.cached:
            self.cached[validationsplit] = self._apply_validation_split(validationsplit)
        
        Rtrain, Rtest = self.cached[validationsplit]
        traincount, testcount = (np.sum(Rtrain), np.sum(Rtest))
        
        if verbose:
            print("learning-rate={0}, regularization-lambda={1}, iterations={2}, validationsplit={3}".format(learningrate, regularization, iterations, validationsplit))
            print("{0} for train, {1} for test, total {2} ratings".format(traincount,testcount,traincount+testcount))
        
        history = []
        for it in range(int(iterations)):
            t = it+1 # t = 1-based iteration index
                        
            # forward propagation overall
            E = (np.dot(self.W[self.ITEM],self.W[self.USER].T)-self.Y)
            
            # test validation error
            Etest = E*Rtest
            MSEtest = np.sum(np.square(Etest))/2
            
            # train & gradient update by ADAM optimizer
            Etrain = E*Rtrain
            MSEtrain = np.sum(np.square(Etrain))/2
            
            # gradient for both W[ITEM] and W[USER]
            G = [
                np.dot(Etrain,self.W[self.USER]) + regularization*self.W[self.ITEM],
                np.dot(Etrain.T,self.W[self.ITEM]) + regularization*self.W[self.USER]
            ]
            for i in (self.ITEM, self.USER):
                self.M1[i] = beta1*self.M1[i] + (1-beta1)*G[i]
                self.M2[i] = beta2*self.M2[i] + (1-beta2)*np.square(G[i])
                MU1 = self.M1[i]/(1-beta1**t)
                MU2 = self.M2[i]/(1-beta2**t)
                self.W[i] -= learningrate*MU1/(np.sqrt(MU2)+1e-7)
            
            # train and test loss
            Ltrain, Ltest = (MSEtrain/traincount, MSEtest/testcount if testcount > 0 else None)
            
            # verbose output
            if verbose and (t == 1 or t % (iterations//10) == 0):
                print("iteration #{0}: loss={1}{2}".format(t, Ltrain, ", valloss={}".format(Ltest) if Ltest else ""))
            history.append({'trainloss': Ltrain, 'testloss': Ltest})
        return history
    
    def instant_fit(self, side, ratings, learningrate=3e-4, regularization=1.0, iterations=1e4, beta1=0.9, beta2=0.999, namedout=True, verbose=False):
        opp = 0 if side else 1
        iconv = self.rateinvconv
        namemap1, namemap2 = (self.namemap[side] if namedout else self.srcmap[side], self.namemap[opp] if namedout else self.srcmap[opp])
        y,r = [0.0]*self.shape[opp], [0.0]*self.shape[opp]
        m1,m2,u1,u2 = [0.0]*self.n, [0.0]*self.n, [0.0]*self.n, [0.0]*self.n
        src = [ (self.seqmap[opp][rel[0]],self.rateconv(rel[1])) for rel in ratings]
        for rel in src:
            y[rel[0]] = rel[1]
            r[rel[0]] = 1            
        w = np.fabs(np.random.normal(0,1/math.sqrt(self.n),self.n)) # single weight vector, one of item or user
        for it in range(int(iterations)):
            t = it+1 # t = 1-based iteration index
            # forward propagation overall
            mse = 0
            g = [0.0]*self.n
            for seq, value in src:
                dot = 0.0
                for i in range(self.n):
                    dot += self.W[opp][seq][i]*w[i]
                e = (dot-value)
                for i in range(self.n):
                    g[i] += e*self.W[opp][seq][i]
                mse += e**2
            mse /= 2

            for i in range(self.n):
                g[i] += regularization*w[i]
                m1[i] = beta1*m1[i] + (1-beta1)*g[i]
                m2[i] = beta2*m2[i] + (1-beta2)*(g[i]**2)
                u1[i] = m1[i]/(1-beta1**t)
                u2[i] = m2[i]/(1-beta2**t)
                w[i] -= learningrate*u1[i]/(math.sqrt(u2[i])+1e-7)

            if verbose and (t == 1 or t % (iterations//10) == 0):
                print("iteration #{0}: loss={1}".format(t,mse/len(src)))
        preds = []
        for seq, pval in enumerate(np.dot(w, self.W[opp].T)):
            preds.append((namemap2[seq], round(iconv(pval),2), iconv(y[seq]) if r[seq] else None))
        return {'weight':w, 'prediction':preds}
    
    def predict(self, side, target, namedout=True):
        if side not in (self.ITEM, self.USER):
            raise ValueError("side can be either ITEM(0) or USER(1): {}".format(side))
        opp, Y, R, iconv = (0 if side else 1, self.Y.T if side else self.Y, self.R.T if side else self.R, self.rateinvconv)
        namemap1, namemap2 = (self.namemap[side] if namedout else self.srcmap[side], self.namemap[opp] if namedout else self.srcmap[opp])
        targetrange = [target] if not isinstance(target, list) and target in self.seqmap[side] else target
        targetseq = [self.seqmap[side][t] for t in targetrange]
        predresult = np.dot(self.W[side][targetseq,:], self.W[opp].T)
        return {
            namemap1[s1]:
            [(namemap2[s2], round(iconv(v),2), iconv(Y[s1,s2]) if R[s1,s2] else None) for s2, v in enumerate(predresult[i])]
            for i,s1 in enumerate(targetseq)            
        }