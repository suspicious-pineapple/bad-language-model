
class NeuralNetwork {
    constructor(shape){
        this.shape = shape;
        this.dtype = Float32Array;
        
        // Calculate total weight size for flat storage
        let totalWeights = 0;
        let totalBiases = 0;
        for(let i = 1; i < shape.length; i++){
            totalWeights += shape[i-1] * shape[i];
            totalBiases += shape[i];
        }

        // Flat storage for weights and biases for cache efficiency
        this.weightsFlat = new this.dtype(totalWeights);
        this.biasesFlat = new this.dtype(totalBiases);
        
        // Offsets to find layer data in flat arrays
        this.weightOffsets = new Int32Array(shape.length);
        this.biasOffsets = new Int32Array(shape.length);
        
        let wOffset = 0;
        let bOffset = 0;
        for(let i = 1; i < shape.length; i++){
            this.weightOffsets[i] = wOffset;
            this.biasOffsets[i] = bOffset;
            wOffset += shape[i-1] * shape[i];
            bOffset += shape[i];
        }
        
        // Input layer values
        this.values = [];
        this.values[0] = new this.dtype(shape[0]);
        
        // Output layer values
        for(let i = 1; i < shape.length; i++){
            this.values[i] = new this.dtype(shape[i]);
        }
    }

    runAll(input){
        const shape = this.shape;
        const weightsFlat = this.weightsFlat;
        const biasesFlat = this.biasesFlat;
        const values = this.values;
        const len = shape.length;
        
        // Use a temporary buffer for current input to avoid modifying input if it's shared, 
        // but typically runAll expects input to be the first layer.
        // We assume input is a Float32Array or similar.
        let current = input;
        
        for(let i = 1; i < len; i++){
            const output = values[i];
            const numOutputs = shape[i];
            const numInputs = shape[i-1];
            
            const wOffset = this.weightOffsets[i];
            const bOffset = this.biasOffsets[i];
            
            // Reset output to biases
            for(let j = 0; j < numOutputs; j++){
                output[j] = biasesFlat[bOffset + j];
            }

            // Matrix-vector multiplication
            // Optimized: Iterate over outputs, then inputs
            // Weights are stored row-major: weights[output_idx * numInputs + input_idx]
            // Actually, in the original code: weights[i][j] is the weight vector for output j.
            // So weightsFlat index for output j, input k is: wOffset + j * numInputs + k
            
            for(let j = 0; j < numOutputs; j++){
                let sum = 0;
                const wBase = wOffset + j * numInputs;
                for(let k = 0; k < numInputs; k++){
                    sum += weightsFlat[wBase + k] * current[k];
                }
                let val = output[j] + sum;
                // Leaky ReLU
                if(val < 0) {
                    val *= 0.1;
                }
                output[j] = val;
            }
            
            current = output;
        }
        return current;
    }

    runLayer(i, input){
        const output = this.values[i];
        const weightsFlat = this.weightsFlat;
        const biasesFlat = this.biasesFlat;
        const numOutputs = this.shape[i];
        const numInputs = this.shape[i-1];
        
        const wOffset = this.weightOffsets[i];
        const bOffset = this.biasOffsets[i];
        
        // Reset output to biases
        for(let j = 0; j < numOutputs; j++){
            output[j] = biasesFlat[bOffset + j];
        }

        // Matrix-vector multiplication
        for(let j = 0; j < numOutputs; j++){
            let sum = 0;
            const wBase = wOffset + j * numInputs;
            for(let k = 0; k < numInputs; k++){
                sum += weightsFlat[wBase + k] * input[k];
            }
            let val = output[j] + sum;
            // Leaky ReLU activation
            if(val < 0) {
                val *= 0.1;
            }
            output[j] = val;
        }
            
        return output;
    }

    serialize(){
        const weightsFlat = this.weightsFlat;
        const biasesFlat = this.biasesFlat;
        const shape = this.shape;
        
        // Estimate total size
        let totalSize = 0;
        for(let i = 1; i < shape.length; i++){
            totalSize += shape[i] * (1 + shape[i-1]); // 1 bias + numInputs weights per output neuron
        }
        
        const result = new Array(totalSize);
        let idx = 0;
        
        for(let i = 1; i < shape.length; i++){
            const numOutputs = shape[i];
            const numInputs = shape[i-1];
            const bOffset = this.biasOffsets[i];
            const wOffset = this.weightOffsets[i];
            
            for(let j = 0; j < numOutputs; j++){
                result[idx++] = biasesFlat[bOffset + j];
                const wBase = wOffset + j * numInputs;
                for(let k = 0; k < numInputs; k++){
                    result[idx++] = weightsFlat[wBase + k];
                }
            }
        }
        return result;
    }
    
    deserialize(weightsExport){
        const weightsFlat = this.weightsFlat;
        const biasesFlat = this.biasesFlat;
        const shape = this.shape;
        
        let idx = 0;
        const lenExport = weightsExport.length;
        
        for(let i = 1; i < shape.length; i++){
            const numOutputs = shape[i];
            const numInputs = shape[i-1];
            const bOffset = this.biasOffsets[i];
            const wOffset = this.weightOffsets[i];
            
            for(let j = 0; j < numOutputs; j++){
                if(idx < lenExport) biasesFlat[bOffset + j] = weightsExport[idx++];
                const wBase = wOffset + j * numInputs;
                for(let k = 0; k < numInputs; k++){
                    if(idx < lenExport) weightsFlat[wBase + k] = weightsExport[idx++];
                }
            }
        }
        return weightsExport;
    }
}

class Optimizer {
    constructor(evaluate, size, hparamsGiven, improvementCallback = () => {}){
        this.hparams = hparamsGiven;
        this.size = size;
        this.improvementCallback = improvementCallback;
        
        // Use Float32Array for numerical performance
        this.dtype = Float32Array;
        this.currentBest = new this.dtype(size);
        // Initialize to 0 is default for typed arrays
        
        this.currentBestLoss = 9999;
        
        // Pre-allocate buffers for training indices and directions
        const n_idx = hparamsGiven.n_idx || 16;
        this.trainIdx = new Int32Array(n_idx); 
        this.direction = new Float32Array(n_idx);
        
        this.directionSign = 1;
        this.lastWasGood = false;
        this.streak = 0;
        this.evaluate = evaluate;
        this.iteration = 0;
        this.lastReportedLoss = 99999;
        
        // Reuse candidate array
        this.candidate = new this.dtype(size);
        
        // Simple LCG state for faster random generation if needed, 
        // but sticking to Math.random for compatibility unless we implement our own.
        // We'll just optimize the loop.
    }

    step(){
        this.iteration++;
        
        // Copy currentBest to candidate
        const currentBest = this.currentBest;
        const candidate = this.candidate;
        const size = this.size;
        
        // Use set for fast copy
        candidate.set(currentBest);

        const hparams = this.hparams;
        const n_idx = hparams.n_idx;
        const lr = hparams.lr;
        const frequency = hparams.frequency;
        const iteration = this.iteration;
        
        if(!this.lastWasGood){
            // Generate random indices and directions
            for(let i = 0; i < n_idx; i++){
                this.trainIdx[i] = Math.floor(Math.random() * size);
                this.direction[i] = (Math.random() * 2 - 1) * lr * Math.cos(iteration * frequency);
            }
        }
        
        // Apply perturbations
        for(let i = 0; i < n_idx; i++){
            const idx = this.trainIdx[i];
            let dir = this.direction[i];
            candidate[idx] += dir;
            if(this.lastWasGood){
                candidate[idx] += dir * this.streak * this.streak;
            }
        }
        
        let newLoss = this.evaluate(candidate, false);
        
        if(newLoss < this.currentBestLoss){
            if(this.lastReportedLoss > newLoss){
                this.improvementCallback(candidate, newLoss);
                this.lastReportedLoss = newLoss;
            }

            // Update currentBest
            currentBest.set(candidate);
            this.currentBestLoss = newLoss;
            this.lastWasGood = true;
            this.streak++;
        } else {
            this.lastWasGood = false;
            this.streak = 0;
        }
    }

    train(n){
        if(n === 0){
            while(true){
                this.step();
            }
        }
        for(let i = 0; i < n; i++){
            this.step();
        }
    }
}

let hparams = { 
    frequency: 0.1,
    lr: 0.1,
    n_idx: 16,
};
//EVAL SECTION
let vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-*/,.! ".split("")

let testnet = new NeuralNetwork([vocabulary.length,16,2,16,vocabulary.length]);
let lossCurve = [];
let optimizer = new Optimizer(evaluateAutoencoder,testnet.serialize().length,hparams,(weights,loss)=>{/*console.log(loss)*/;lossCurve.push(loss)});


let startTime = Date.now();

while(Date.now() - startTime < 10000){
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
