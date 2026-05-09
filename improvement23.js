
class NeuralNetwork {
    constructor(shape){
        this.shape = shape;
        this.weights = [];
        this.biases = [];
        this.values = [];
        this.weightsT = [];
        this.dtype = Float32Array;
        this.values[0] = new this.dtype(shape[0]);
        for(let i = 1; i < shape.length; i++){
            let prevSize = shape[i-1];
            let currSize = shape[i];
            this.weights[i] = new this.dtype(prevSize * currSize);
            this.biases[i] = new this.dtype(currSize);
            this.values[i] = new this.dtype(currSize);
            
            let w = this.weights[i];
            let wT = new this.dtype(prevSize * currSize);
            for(let r = 0; r < prevSize; r++){
                for(let c = 0; c < currSize; c++){
                    wT[c * prevSize + r] = w[r * currSize + c];
                }
            }
            this.weightsT[i] = wT;
        }
    }
    runAll(input){
        let current = input;
        for(let i = 1; i < this.shape.length; i++){
            current = this.runLayer(i, current);
        }
        return current;
    }
    runLayer(i, input){
        let outSize = this.shape[i];
        let inSize = this.shape[i-1];
        let weights = this.weightsT[i];
        let biases = this.biases[i];
        let output = this.values[i];
        
        let wStride = inSize;
        let kLimit = inSize - (inSize % 8);
        
        for(let j = 0; j < outSize; j++){
            let sum = biases[j];
            let wBase = j * wStride;
            
            let k = 0;
            while(k < kLimit){
                sum += weights[wBase + k] * input[k];
                sum += weights[wBase + k + 1] * input[k + 1];
                sum += weights[wBase + k + 2] * input[k + 2];
                sum += weights[wBase + k + 3] * input[k + 3];
                sum += weights[wBase + k + 4] * input[k + 4];
                sum += weights[wBase + k + 5] * input[k + 5];
                sum += weights[wBase + k + 6] * input[k + 6];
                sum += weights[wBase + k + 7] * input[k + 7];
                k += 8;
            }
            
            while(k < inSize){
                sum += weights[wBase + k] * input[k];
                k++;
            }
            
            output[j] = sum > 0 ? sum : sum * 0.2;
        }
        
        return output;
    }
    serialize(){
        let weightsExport = [];
        for(let i = 1; i < this.weights.length; i++){
            weightsExport.push(...this.biases[i]);
            weightsExport.push(...this.weights[i]);
        }
        return weightsExport;
    }
    deserialize(weightsExport){
        let ptr = 0;
        for(let i = 1; i < this.weights.length; i++){
            let biasSize = this.biases[i].length;
            this.biases[i].set(weightsExport.slice(ptr, ptr + biasSize));
            ptr += biasSize;
            let weightSize = this.weights[i].length;
            this.weights[i].set(weightsExport.slice(ptr, ptr + weightSize));
            ptr += weightSize;
        }
        this._updateWeightsT();
        return weightsExport;
    }
    _updateWeightsT(){
        for(let i = 1; i < this.weights.length; i++){
            let w = this.weights[i];
            let wT = this.weightsT[i];
            let inSize = this.shape[i-1];
            let outSize = this.shape[i];
            for(let r = 0; r < inSize; r++){
                for(let c = 0; c < outSize; c++){
                    wT[c * inSize + r] = w[r * outSize + c];
                }
            }
        }
    }
}

class Optimizer {
    constructor(evaluate, size, hparamsGiven, improvementCallback = () => {}){
        this.hparams = hparamsGiven;
        this.size = size;
        this.improvementCallback = improvementCallback;
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
        
        this.nIdx = hparamsGiven.n_idx;
        this.lr = hparamsGiven.lr;
    }
    step(){
        this.iteration++;
        
        if (!this.lastWasGood) {
            let nIdx = this.nIdx;
            let size = this.size;
            let lr = this.lr;
            
            this.trainIdx.length = nIdx;
            for(let i = 0; i < nIdx; i++){
                this.trainIdx[i] = Math.floor(Math.random() * size);
            }
            
            this.direction.length = nIdx;
            for(let i = 0; i < nIdx; i++){
                this.direction[i] = (Math.random() * 2 - 1) * lr;
            }
        }
        
        let streakSq = this.streak * this.streak;
        let lastWasGood = this.lastWasGood;
        
        let currentBest = this.currentBest;
        let trainIdx = this.trainIdx;
        let direction = this.direction;
        let nIdx = trainIdx.length;
        
        for(let i = 0; i < nIdx; i++){
            let idx = trainIdx[i];
            let dir = direction[i];
            currentBest[idx] += dir;
            if (lastWasGood) {
                currentBest[idx] += dir * streakSq;
            }
        }
        
        let newLoss = this.evaluate(this.currentBest, false);
        if (newLoss < this.currentBestLoss) {
            if (this.lastReportedLoss > newLoss) {
                this.improvementCallback(this.currentBest, newLoss);
                this.lastReportedLoss = newLoss;
            }
            this.currentBestLoss = newLoss;
            this.lastWasGood = true;
            this.streak++;
        } else {
            for(let i = 0; i < nIdx; i++){
                let idx = trainIdx[i];
                let dir = direction[i];
                currentBest[idx] -= dir;
                if (lastWasGood) {
                    currentBest[idx] -= dir * streakSq;
                }
            }
            this.lastWasGood = false;
            this.streak = 0;
        }
    }
    train(n){
        if (n === 0) {
            while (true) {
                this.step();
            }
        }
        for(let i = 0; i < n; i++){
            this.step();
        }
    }
}

let hparams = {
    frequency: 0.2,
    lr: 0.015,
    n_idx: 50,
};

//EVAL SECTION

let vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-*/,.! ".split("")

let testnet = new NeuralNetwork([vocabulary.length,16,2,16,vocabulary.length]);
let lossCurve = [];
let optimizer = new Optimizer(evaluateAutoencoder,testnet.serialize().length,hparams,(weights,loss)=>{/*console.log(loss)*/;lossCurve.push(loss)});


let startTime = Date.now();

while(Date.now() - startTime < 5000){
    optimizer.train(1000);
    
}

let timeTaken = Date.now() - startTime;

console.log(Math.min(...lossCurve));
function evaluateAutoencoder(weights,celebrate=false){
    let net = new NeuralNetwork(testnet.shape);
    net.deserialize([...weights]);
    let totalLoss = 0;
    let randomSequence = new Float32Array(vocabulary.length);
    for(let i=0;i<vocabulary.length;i++){
        randomSequence.fill(0);
        randomSequence[i]=1;
        let res = net.runAll(randomSequence);
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


function sequenceLoss(seq2,seq1){
    let loss = 0;
    for(let i = 0; i < seq1.length;i++){
        loss+=Math.pow((seq1[i]-seq2[i]),2);
    }

    return loss;
}
