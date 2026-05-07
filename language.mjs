

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
        this.dtype = Float16Array;
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
    forward(inputs){

        this.values[0]=inputs;
        for(let i = 1; i < this.shape.length;i++){
            this.values[i].set(this.biases[i]);
            for(let j = 0; j < this.weights[i].length;j++){
                for(let k = 0; k < this.weights[i][j].length;k++){
                    this.values[i][j] += this.weights[i][j][k]*this.values[i-1][k];
                }
            }
            
            this.values[i]=this.values[i].map(v=>Math.max(v,v*0.06));

            //let min = Math.min(...this.values[i]);
            //let max = Math.max(...this.values[i]);
            //this.values[i] = this.values[i].map(v=>((v-min)/(max-min)));

        }
        return this.values[this.shape.length-1];
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
        let output = new Float16Array(this.shape[i]);
        output.set(this.biases[i]);
            for(let j = 0; j < this.weights[i].length;j++){
                for(let k = 0; k < this.weights[i][j].length;k++){
                    output[j] += this.weights[i][j][k]*input[k];
                }
            }
            output=output.map(v=>Math.max(v,v*0.06));

            return output;

    }


    

    randomize(wlr,blr){
        for(let i = 1; i < this.shape.length;i++){
            
            for(let j = 0; j < this.weights[i].length;j++){
                for(let k = 0; k < this.weights[i][j].length;k++){
                    this.weights[i][j][k]+=(Math.random()*2-1)*wlr
                }
                
                this.biases[i][j]=+(Math.random()*2-1)*blr;
            }
        }
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


let testnet = new NeuralNetwork([vocabulary.length,24,2,8,vocabulary.length]);
//console.log(JSON.stringify(testnet));



testnet.deserialize(testnet.serialize());




let currentBest = testnet;
let trainIdx = Array.from({length:1024},()=>Math.floor(Math.random()*currentBest.serialize().length));
let currentBestScore = evaluateAutoencoder(testnet.serialize());
console.log(currentBestScore);
let iteration = 0;
let lastWasGood = false;
//let weightDiff = new Float16Array(currentBest.serialize().length);
let weightDiff = new Float16Array(trainIdx.length);

class Optimizer {
    constructor(evaluate,size,improvementCallback=()=>{}){
        this.hparams = {
        };
        this.improvementCallback = improvementCallback;
        this.dtype = Array;
        this.currentBest = new Array(size);
        console.log(size);
        this.currentBest.fill(0.1);
        //this.currentBestScore = evaluate(this.currentBest);
       this.currentBestScore = 9999;
        
        this.evaluate = evaluate;
        this.iteration = 0;    
    }
    step(){
        this.iteration++;
        //let candidate = JSON.parse(JSON.stringify(this.currentBest));
        //let candidate = JSON.parse(JSON.stringify(this.currentBest));
        let candidate = [...this.currentBest];
        
         
        for(let i = 0; i < this.iteration%400+500;i++){
            
            let index = Math.floor(Math.random()*candidate.length);
            candidate[index]+= (Math.random()*2-1)*Math.cos(this.iteration*0.001)*0.8;
        }

        let newScore = this.evaluate(candidate,false);
        if(newScore<this.currentBestScore){
            this.improvementCallback(candidate,newScore);
            //console.log("new best: ",newScore);
            this.currentBest=candidate;
            this.currentBestScore=newScore;
            //this.evaluate([...candidate],true);

        }

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
    let randomSequence = new Float16Array(96);
    for(let i=0;i<96;i++){
        randomSequence.fill(0);
        randomSequence[i]=1;
        let res = net.forward(randomSequence,0,2);
        let max = Math.max(...res);
        let maxIndex = res.indexOf(max);
        if(maxIndex==i){
            continue;
        }
        totalLoss+=sequenceLoss(res,randomSequence)*0.5;
        //totalLoss+=sequenceLoss(net.forward(randomSequence),randomSequence);
        
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

function tokenize(text,returnVec=false){
    let parts = text.split("");
    let output = [];
    let decodedStr = "";
    for(let i = 0; i < parts.length;i++){
        let tokenID = vocabulary.indexOf(parts[i]);
        let vec = new Float16Array(vocabulary.length);
        vec[tokenID]=1;
        //let encoded = currentBest.forward(vec,0,1);
        let decoded = currentBest.forward(vec,1,2);
        if(returnVec){
            return decoded;
        }

        let max = Math.max(...decoded);
        let maxIndex = decoded.indexOf(max);
        decodedStr+=vocabulary[maxIndex];

    }
    return decodedStr;
}

function encodeText(text,net){
    let parts = text.split("");
    let output = [];
    for(let i = 0; i < text.length;i++){
        let tokenID = vocabulary.indexOf(parts[i]);
        let vec = new Float16Array(vocabulary.length);
        vec[tokenID]=1;
        let encoded = net.runLayer(2,net.runLayer(1,vec));
        output.push(encoded);        
    }
    return output;
}
function decodeText(arr,net){
    let text = [];
    for(let i = 0; i < arr.length;i++){
        let res = net.runLayer(4,net.runLayer(3,arr[i]));
        let tokenID = res.indexOf(Math.max(...res));
        text.push(vocabulary[tokenID]);
    }
    return text.join("");
}


//let optim = new Optimizer(evaluateAutoencoder,testnet.serialize().length);

//optim.train(100000);



class ThreadedOptimizer {
    constructor(evaluate,size,numThreads=8){

        this.numThreads = numThreads;
        this.evaluate = evaluate;
        this.size = size;
        this.globalBest = new Array(size);
        this.globalBestScore = 999999999999999; 
        this.workers = [];
        for(let i = 0; i < numThreads;i++){
            let newWorker = new Worker("./language.mjs");
            //et newerWorker = new Worker("./language.mjs");
            let opt = this;
            newWorker.on("message",(msg)=>{
                let data = JSON.parse(msg);
                if(data.score < opt.globalBestScore){
                    opt.globalBest = data.weights;
                    console.log("new best! ",data.score);
                    console.log("from worker ",i)
                    console.log(evaluate([...data.weights],true));
                    
                    opt.workers.forEach(w=>{
                        w.postMessage(msg);
                    })
                }
            })
            this.workers.push(newWorker);

        }
    }
}

if(isMainThread){
    let optimizer = new ThreadedOptimizer(evaluateAutoencoder,testnet.serialize().length,10);
    
} else {
    console.log(parentPort)
    
    try {
    let incoming = null;
    let optimizer = new Optimizer(evaluateAutoencoder,testnet.serialize().length,(weights,score)=>{
        
        let msg = JSON.stringify({weights,score});
        parentPort.postMessage(msg);
        });
    
    parentPort.on("message",msg=>{
        let data = JSON.parse(msg);
        incoming = data;
        //optimizer.currentBest = msg.weights;
        //optimizer.currentBestScore = msg.score;
    });
    
    
    while(true){
        await new Promise(resolve=>setTimeout(resolve,500));
        optimizer.train(100);
        if(incoming !=null){
            optimizer.currentBestScore=incoming.score;
            optimizer.currentBest=incoming.weights;
            incoming=null;
        }

    }
    } catch(e){
        console.log(e)
    }
}























