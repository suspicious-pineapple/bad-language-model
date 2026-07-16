

import {
Worker, isMainThread,parentPort,workerData
} from "node:worker_threads";

let vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-*/,.! ".split("")

class Optimizer {
    constructor(evaluate, size, hparamsGiven, improvementCallback = () => {}) {
        this.hparams = hparamsGiven;
        this.size = size;
        this.improvementCallback = improvementCallback;
        this.dtype = Array;
        
        // State for persistent exploration
        this.currentBest = new Array(size);
        this.currentBest.fill(0);
        this.currentBestLoss = 9999;

        this.trainIdx = [];
        this.direction = [];
        this.directionSign = 1;
        this.lastWasGood = false;

        this.streak = 0;
        this.evaluate = evaluate;
        this.iteration = 0;

        this.lastReportedLoss = 99999;
        this.exploreBuffer = [];
        this.exploreCount = 0;
        
        // Internal state for gradient approximation
        this.gradientEstimate = new Array(size).fill(0);
        this.gradientHistory = [];
        this.noiseSamples = [];
        this.maxSamples = 10; // Keep history for momentum
        
        // Momentum accumulators for each parameter
        this.velocity = new Array(size).fill(0);
        this.acc = new Array(size).fill(0); // For Adam-like second moment
        
        // Persistent direction vector to maintain coherence
        this.persistentDirection = new Array(size).fill(0);
        this.lastUpdate = new Array(size).fill(0);
        
        // Cache for loss evaluation to avoid redundant calls
        this.cachedLoss = null;
        this.cachedCandidate = null;
    }

    step() {
        this.iteration++;
        let candidate = [...this.currentBest];
        
        // Determine if we are in a persistent exploration phase
        // Reassign train_idx every few iterations to maintain coherence in direction
        // Increased interval to allow for more aggressive momentum building
        const reassignInterval = Math.max(4, Math.floor(this.hparams.n_idx / 2));
        const shouldReassign = this.iteration % reassignInterval === 0;

        if (!this.lastWasGood || shouldReassign) {
            // Aggressive initial exploratory phase: larger random steps
            // If we just failed, we are in exploration mode.
            // If we are in a new interval, we might be exploring or continuing a trend.
            
            this.trainIdx = Array(this.hparams.n_idx);
            for (let i = 0; i < this.trainIdx.length; i++) {
                this.trainIdx[i] = Math.floor(Math.random() * this.size);
            }

            this.direction = [];
            // Aggressive learning rate for exploration
            const exploreFactor = this.lastWasGood ? 1.0 : 4.0; 
            
            for (let i = 0; i < this.trainIdx.length; i++) {
                // Use cosine modulation for smooth oscillation, but scale by exploreFactor
                const baseNoise = (Math.random() * 2 - 1);
                const freqMod = Math.cos(this.iteration * this.hparams.frequency);
                this.direction[i] = baseNoise * this.hparams.lr * exploreFactor * freqMod;
            }
            
            // Reset streak if we are forcing a reassignment or just failed
            if (!this.lastWasGood) {
                this.streak = 0;
            }
        }

        // Apply updates with momentum and gradient approximation
        for (let i = 0; i < this.trainIdx.length; i++) {
            const idx = this.trainIdx[i];
            let update = this.direction[i];
            
            // Gradient Approximation using Finite Differences
            // We approximate the gradient by comparing the current loss with the loss 
            // if we had moved in the opposite direction (conceptually). 
            // However, since we only have one forward pass per step in this loop, 
            // we use the history of improvements to build a running estimate.
            
            // If we had a good step previously, amplify the direction (momentum-like)
            // but cap it to prevent explosion
            if (this.lastWasGood && this.streak > 0) {
                const momentum = Math.min(this.streak * 0.2, 1.5); 
                update *= momentum;
            }
            
            // Apply Adam-like normalization to the update
            // This helps stabilize the learning rate across different scales of parameters
            const beta1 = 0.9;
            const beta2 = 0.999;
            const epsilon = 1e-8;
            
            this.acc[idx] = beta1 * this.acc[idx] + (1 - beta1) * update;
            this.velocity[idx] = beta2 * this.velocity[idx] + (1 - beta2) * update * update;
            
            const biasCorrection1 = 1 - Math.pow(beta1, this.iteration);
            const biasCorrection2 = 1 - Math.pow(beta2, this.iteration);
            
            const m = this.acc[idx] / biasCorrection1;
            const v = this.velocity[idx] / biasCorrection2;
            
            const normalizedUpdate = m / (Math.sqrt(v) + epsilon);
            
            // Scale back by the learning rate
            candidate[idx] += normalizedUpdate * this.hparams.lr;
        }

        let newLoss = this.evaluate(candidate, false);
        
        if (newLoss < this.currentBestLoss) {
            if (this.lastReportedLoss > newLoss) {
                this.improvementCallback(candidate, newLoss);
                this.lastReportedLoss = newLoss;
            }

            this.currentBest = candidate;
            this.currentBestLoss = newLoss;
            
            if (this.lastWasGood) {
                // Consistent improvement: increase streak
                this.streak++;
            } else {
                // First improvement after failure: reset streak to 1
                this.streak = 1;
            }
            this.lastWasGood = true;
        } else {
            this.lastWasGood = false;
            this.streak = 0;
        }
    }

    train(n) {
        if (n == 0) {
            while (true) {
                this.step();
            }
        }
        for (let i = 0; i < n; i++) {
            this.step();
        }
    }
}

let hparams = { //these are the default hparams that get used when the optimizer is initialized
    frequency: 0.1,
    lr: 0.2,
    n_idx: 16,
};


class NeuralNetwork {
    constructor(shape) {
        this.shape = shape;
        this.weights = [];
        this.biases = [];
        this.values = [];
        this.dtype = Float32Array;
        this.values[0] = new this.dtype(shape[0]);
        for (let i = 1; i < shape.length; i++) {
            this.weights[i] = []; //todo: fuse?
            for (let j = 0; j < this.shape[i]; j++) {
                this.weights[i][j] = new this.dtype(shape[i - 1]);
            }
            this.biases[i] = new this.dtype(shape[i]);
            this.values[i] = new this.dtype(shape[i]);
        }
    }
    runAll(input) {
        let current = input;
        for (let i = 1; i < this.shape.length; i++) {
            current = this.runLayer(i, current);
        }
        return current;
    }

    runLayer(i, input) {
        //this.values[i].set(input);
        //let output = new this.dtype(this.shape[i]);
        let output = this.values[i].fill(0);
        output.set(this.biases[i]);
        for (let j = 0; j < this.weights[i].length; j++) {
            for (let k = 0; k < this.weights[i][j].length; k++) {
                output[j] += this.weights[i][j][k] * input[k];
            }
            output[j] = output[j] < 0 ? output[j] * 0.1 : output[j];
        }


        return output;

    }


    serialize() {
        let weightsExport = [];
        for (let i = 1; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                weightsExport.push(this.biases[i][j]);
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    weightsExport.push(this.weights[i][j][k]);
                }
            }
        }
        return weightsExport;
    }
    deserialize(weightsExport) {
        for (let i = 1; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                this.biases[i][j] = weightsExport.shift();
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] = weightsExport.shift();
                }
            }
        }
        return weightsExport;
    }

}



let testnet = new NeuralNetwork([vocabulary.length,vocabulary.length,vocabulary.length]);
//console.log(JSON.stringify(testnet));







testnet.deserialize(testnet.serialize());
let currentBest = testnet;
let currentBestLoss = evaluateAutoencoder(testnet.serialize());
console.log(currentBestLoss);



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
                if(data.loss < opt.globalBestLoss){
                    if(Math.random()>0.7){
                    
                        for(let j = 0; j < opt.globalBest.length;j++){
                            if(Math.random()>0.5){
                                data.weights[j]=opt.globalBest[j];
                            }
                        }    
                        
                        
                    }
                    opt.globalBest = data.weights;

                    let improvement = opt.globalBestLoss-data.loss;
                    opt.globalBestLoss = data.loss;
                    opt.workerHparams[i]=data.hparams;
                    console.log("new best! ",data.loss);
                    console.log("from worker ",i);
                    
                    console.log(evaluate([...data.weights],true));
                    if(data.loss < 35 && !opt.hasCelebrated){
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
    let optimizer = new ThreadedOptimizer(evaluateAutoencoder,testnet.serialize().length,10);
    
} else {
    console.log(parentPort)
    
    try {
    let incoming = null;
    let hparams = {
        frequency:0.2,
        lr:0.015,
        n_idx:50,
    };
    let optimizer = new Optimizer(evaluateAutoencoder,testnet.serialize().length,hparams,(weights,loss)=>{
        
        let msg = JSON.stringify({weights,loss,hparams:optimizer.hparams});
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
        

        
        if(incoming !=null && incoming.loss<optimizer.currentBestLoss){
            optimizer.currentBestLoss=incoming.loss;
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























