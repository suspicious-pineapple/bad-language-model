


let vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-*/,.! ".split("")



class NeuralNetwork {
    constructor(shape){
        this.shape = shape;
        this.weights = [];
        this.biases = [];
        this.values = [];
        this.dtype = Float16Array;
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
                    //console.log(i,j,k,this.weights[i][j][k],this.values[i-1][k])
                }
            }
            //this.values[i]=this.values[i].map(v=>Math.max(v*0.05,v));
            this.values[i]=this.values[i].map(v=>Math.max(v,v*0.06));

            //let min = Math.min(...this.values[i]);
            //let max = Math.max(...this.values[i]);
            //this.values[i] = this.values[i].map(v=>((v-min)/(max-min)));

        }
        return this.values[this.shape.length-1];
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


let testnet = new NeuralNetwork([70,24,2,8,70]);
//console.log(JSON.stringify(testnet));
//testnet.randomize(1,0.1);
testnet.deserialize(testnet.serialize());




let currentBest = testnet;
let trainIdx = Array.from({length:1024},()=>Math.floor(Math.random()*currentBest.serialize().length));
let currentBestScore = evaluateAutoencoder(testnet);
console.log(currentBestScore);
let iteration = 0;
let lastWasGood = false;
//let weightDiff = new Float16Array(currentBest.serialize().length);
let weightDiff = new Float16Array(trainIdx.length);
while(true){
    iteration++;

    let wlr = (Math.sin(iteration*0.005)+0.5)*0.05;
    if(iteration%512==0){
        trainIdx = Array.from({length:Math.ceil(Math.random()*512)},()=>Math.floor(Math.random()*currentBest.serialize().length));
    }
    if(iteration%1000==0){
        console.log("iteration ",iteration);
        //console.log("wlr ",wlr);

    }
    let blr = wlr;
    let candidate = new NeuralNetwork([70,24,2,8,70]);
    let newWeights = currentBest.serialize();
    weightDiff = new Float16Array(trainIdx.length);

    if(!lastWasGood){
        for(let i = 0; i < trainIdx.length;i++){
            //weightDiff.fill(0);
            weightDiff[i]=(Math.random()*2-1)*wlr
        }
        
    }
    //newWeights = newWeights.map((v,i)=>newWeights[i]+(weightDiff[i]*0.0001));
    for(let i = 0; i < trainIdx.length;i++){
        newWeights[trainIdx[i]]+=weightDiff[i]*0.1
    }
    
    
    
    

    candidate.deserialize(newWeights);
    
    let newScore = evaluateAutoencoder(candidate);
        if(newScore<currentBestScore){
            if(lastWasGood){
                console.log("reused change!");
            }
        lastWasGood=true;
        currentBest=candidate;
        currentBestScore=newScore;
        console.log("new best:",newScore);
        let evalArr = new Float16Array(candidate.shape[0]);
        evalArr[2]=1;
        //console.log(candidate.forward(evalArr));
        console.log(tokenize("the quick brown frog jumps over the lazy fox"));
        if(tokenize("is it good now")=="is it good now"){
            console.log(tokenize("abcdefghijklmnopqrstuvwxyz"));

        }
    } else {
        lastWasGood=false;
    }
}




function evaluateAutoencoder(net){
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
        let vec = new Float16Array(70);
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































