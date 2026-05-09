

import { resolve } from "node:dns";
import {
Worker, isMainThread,parentPort,workerData
} from "node:worker_threads";

let vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-*/,.! ".split("")



class NeuralNetwork {
    constructor(shape){
        this.shape = shape;
        this.weights = [];
        this.biases = [];
        this.values = [];
        this.dtype = Float32Array;
        this.values[0] = new this.dtype(shape[0]);
        for(let i = 1; i < shape.length;i++){
            this.weights[i] = []; //todo: fuse?
            for(let j = 0; j < this.shape[i];j++){
                this.weights[i][j] = new this.dtype(shape[i-1]);
            }
            this.biases[i] = new this.dtype(shape[i]);
            this.values[i] = new this.dtype(shape[i]);
        }

    }
    runAll(input){
        let current = input;
        for(let i = 1; i < this.shape.length;i++){
            current = this.runLayer(i,current);
        }
        return current;
    }

    runLayer(i,input){
        
        //this.values[i].set(input);
        //let output = new this.dtype(this.shape[i]);
        let output = this.values[i].fill(0);
        output.set(this.biases[i]);
            for(let j = 0; j < this.weights[i].length;j++){
                for(let k = 0; k < this.weights[i][j].length;k++){
                    output[j] += this.weights[i][j][k]*input[k];
                }
            }
            output=output.map(v=>Math.max(v,v*0.2));
            //output=output.map(v=>Math.tanh(v));

            return output;

    }


    serialize(){
        let weightsExport = [];
        for(let i = 1; i < this.weights.length;i++){
            for(let j = 0; j < this.weights[i].length;j++){
                weightsExport.push(this.biases[i][j]);
                for(let k = 0; k < this.weights[i][j].length;k++){
                    weightsExport.push(this.weights[i][j][k]);
                }
            }
        }
        return weightsExport;
    }
    deserialize(weightsExport){
        for(let i = 1; i < this.weights.length;i++){
            for(let j = 0; j < this.weights[i].length;j++){
                this.biases[i][j] = weightsExport.shift();
                for(let k = 0; k < this.weights[i][j].length;k++){
                    this.weights[i][j][k] = weightsExport.shift();
                }
            }
        }
        return weightsExport;
    }
    
}


let testnet = new NeuralNetwork([vocabulary.length,2,vocabulary.length]);
//console.log(JSON.stringify(testnet));



testnet.deserialize(testnet.serialize());




let currentBest = testnet;
let currentBestLoss = evaluateAutoencoder(testnet.serialize());
console.log(currentBestLoss);

class Optimizer {
    constructor(evaluate,size,hparamsGiven,improvementCallback=()=>{}){
        let hparams = {
        };
        Object.entries(hparamsGiven).forEach(e=>{
            hparams[e[0]]=[e[1]];
        });
        this.hparams=hparamsGiven;
        this.size = size;
        this.improvementCallback = improvementCallback;
        this.dtype = Array;
        this.currentBest = new Array(size);
        console.log(size);
        this.currentBest.fill(0);
        //this.currentBestLoss = evaluate(this.currentBest);
        this.currentBestLoss = 9999;
        
        this.trainIdx = [];
        //for(let i=0;i<this.size;i++){

        //        this.trainIdx[i]=i;
            
        //}
        this.direction = [];
        this.directionSign = 1;
        this.lastWasGood = false;


        this.streak = 0;
        this.evaluate = evaluate;
        this.iteration = 0;

        this.lastReportedLoss = 99999;

    }
    step(){
        this.iteration++;
        let candidate = [...this.currentBest];
        
        
        

        if(true||!this.lastWasGood){
            this.trainIdx = Array(this.hparams.n_idx);
            for(let i = 0; i<this.trainIdx.length;i++){
                this.trainIdx[i] = Math.floor(Math.random()*this.size);
            }
            //console.log(this.trainIdx)
            this.direction = [];
            
            
            for(let i = 0; i<this.trainIdx.length;i++){
                this.direction[i]=(Math.random()*2-1)*this.hparams.lr;
            }
        }
        
        for(let i = 0; i < this.trainIdx.length;i++){
            
            //candidate[trainIdx[i]]+= (Math.random()*2-1)*Math.cos(this.iteration*0.001*this.hparams.frequency)*this.hparams.lr;
            candidate[this.trainIdx[i]]+= this.direction[i];
            if(this.lastWasGood){
                candidate[this.trainIdx[i]]+= this.direction[i]*this.streak*this.streak;
            }
        }
        
        

        
        let newLoss = this.evaluate(candidate,false);
        if(newLoss<this.currentBestLoss){
            
            if(this.lastReportedLoss*1 > newLoss){
                this.improvementCallback(candidate,newLoss);
                this.lastReportedLoss = newLoss;
            }

            this.currentBest=candidate;
            this.currentBestLoss=newLoss;
            if(this.lastWasGood){
                //console.log("reused gradient! streak:",this.streak);
            }
            this.lastWasGood=true;
            this.streak++;
        } else {this.lastWasGood=false;this.streak=0};

    }

    train(n){
        if(n==0){
            while(true){
                this.step();
            }
        }
        for(let i = 0; i < n ; i++){
            this.step();
        }
    }

}


function evaluateAutoencoder(weights,celebrate=false){
    let net = new NeuralNetwork(testnet.shape);
    net.deserialize([...weights]);
    let totalLoss = 0;
    let randomSequence = new Float32Array(vocabulary.length);
    for(let i=0;i<96;i++){
        randomSequence.fill(0);
        randomSequence[i]=1;
        let res = net.runAll(randomSequence,0,2);
        let max = Math.max(...res);
        let maxIndex = res.indexOf(max);
        if(maxIndex==i){
            continue;
        }
        totalLoss+=sequenceLoss(res,randomSequence)*0.5;
        
        
    }
    if(celebrate){
        console.log(decodeText(encodeText("yay new best!",net),net));
    }
    return totalLoss;
}



function sequenceLoss(seq1,seq2){
    let loss = 0;
    for(let i = 0; i < seq1.length;i++){
        loss+=Math.pow((seq1[i]-seq2[i]),2);
    }

    return loss;
}

function encodeText(text,net){
    let parts = text.split("");
    let output = [];
    for(let i = 0; i < text.length;i++){
        let tokenID = vocabulary.indexOf(parts[i]);
        let vec = new Float32Array(vocabulary.length);
        vec[tokenID]=1;
        //let encoded = net.runLayer(1,net.runLayer(1,vec));
        let encoded = net.runLayer(1,vec);
        
        output.push(encoded);        
    }
    return output;
}
function decodeText(arr,net){
    let text = [];
    for(let i = 0; i < arr.length;i++){
        let res = net.runLayer(2,arr[i]);
        let tokenID = res.indexOf(Math.max(...res));
        text.push(vocabulary[tokenID]);
    }
    return text.join("");
}


//let optim = new Optimizer(evaluateAutoencoder,testnet.serialize().length);

//optim.train(100000);



class ThreadedOptimizer {
    constructor(evaluate,size,numThreads=6){
        this.startTime = Date.now();
        this.iteration = 0;
        this.numThreads = numThreads;
        this.evaluate = evaluate;
        this.size = size;
        this.globalBest = new Array(size);
        this.globalBestLoss = 999999999999999; 
        this.workers = [];
        this.hasCelebrated = false;
        this.celebrationThreshold = 25;
        this.scoreboard = new Array(numThreads);
        this.workerHparams = new Array(numThreads);
        this.scoreboard.fill(0);
        for(let i = 0; i < numThreads;i++){
            let newWorker = new Worker("./language.mjs");
            //et newerWorker = new Worker("./language.mjs");
            let opt = this;
            newWorker.on("message",(msg)=>{
                let data = JSON.parse(msg);
                if(data.Loss < opt.globalBestLoss){
                    if(Math.random()>0.7){
                    
                        for(let j = 0; j < opt.globalBest.length;j++){
                            if(Math.random()>0.5){
                                data.weights[j]=opt.globalBest[j];
                            }
                        }    
                        
                        
                    }
                    opt.globalBest = data.weights;

                    let improvement = opt.globalBestLoss-data.Loss;
                    opt.globalBestLoss = data.Loss;
                    opt.workerHparams[i]=data.hparams;
                    console.log("new best! ",data.Loss);
                    console.log("from worker ",i);
                    
                    console.log(evaluate([...data.weights],true));
                    if(data.Loss < 35 && !opt.hasCelebrated){
                        console.log("-------");
                        console.log("reaching loss 35 took ",(Date.now()-opt.startTime)/1000);
                        console.log("-------");
                        opt.hasCelebrated = true;
                    }
                    opt.scoreboard[i]+=improvement;
                    opt.iteration++;
                    data.hparams = null;
                    if(opt.iteration > 40){
                        let maxImprovements = Math.max(...opt.scoreboard);
                        let bestWorker = opt.scoreboard.indexOf(maxImprovements);
                        console.log("scoreboard:",opt.scoreboard)
                        opt.scoreboard = new Array(opt.numThreads);
                        opt.scoreboard.fill(0);
                        data.hparams = opt.workerHparams[bestWorker];
                        console.log("distributing best hparam set from worker ",bestWorker," params:",opt.workerHparams[bestWorker]);
                        opt.iteration=0;
                    }
                    opt.workers.forEach(w=>{
                        w.postMessage(JSON.stringify(data));
                    })
                }
            })
            this.workers.push(newWorker);

        }
    }
}

if(isMainThread){
    let optimizer = new ThreadedOptimizer(evaluateAutoencoder,testnet.serialize().length,1);
    
} else {
    console.log(parentPort)
    
    try {
    let incoming = null;
    let hparams = {
        frequency:0.2,
        lr:0.015,
        n_idx:50,
    };
    let optimizer = new Optimizer(evaluateAutoencoder,testnet.serialize().length,hparams,(weights,Loss)=>{
        
        let msg = JSON.stringify({weights,Loss,hparams:optimizer.hparams});
        parentPort.postMessage(msg);
        });
    
    parentPort.on("message",msg=>{
        let data = JSON.parse(msg);
        incoming = data;

    });
    
    
    while(true){
        await new Promise(resolve=>setTimeout(resolve,Math.random()*5));
        //await new Promise(process.nextTick);
        //console.log("pulse");

        let startTime = Date.now();
        optimizer.train(1000);
        let timeTaken = Date.now() - startTime;
        console.log("time needed:",timeTaken);
        

        
        if(incoming !=null && incoming.Loss<optimizer.currentBestLoss){
            optimizer.currentBestLoss=incoming.Loss;
            optimizer.currentBest=incoming.weights;
            if(incoming.hparams != null){
                console.log("received hparams:",incoming.hparams)
                optimizer.hparams = incoming.hparams;
            }
            optimizer.hparams.frequency += (Math.random()*2-1)*0.01;
            optimizer.hparams.lr = optimizer.hparams.lr * (1+(Math.random()*2-1)*0.05)
            optimizer.hparams.n_idx += Math.round(Math.random()*2-1)*3;
            if(optimizer.hparams.n_idx<=0){
                optimizer.hparams.n_idx=1;
            }
        
            incoming=null;
        }

    }
    } catch(e){
        console.log(e)
    }
}























