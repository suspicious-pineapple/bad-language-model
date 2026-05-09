
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

let hparams = {
    frequency:0.2,
    lr:0.015,
    n_idx:50,
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
