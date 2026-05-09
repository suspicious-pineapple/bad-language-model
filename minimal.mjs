
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
        //this.currentBestScore = evaluate(this.currentBest);
        this.currentBestScore = 9999;
        
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

        this.lastReportedScore = 99999;

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
        
        

        
        let newScore = this.evaluate(candidate,false);
        if(newScore<this.currentBestScore){
            
            if(this.lastReportedScore*1 > newScore){
                this.improvementCallback(candidate,newScore);
                this.lastReportedScore = newScore;
            }

            this.currentBest=candidate;
            this.currentBestScore=newScore;
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