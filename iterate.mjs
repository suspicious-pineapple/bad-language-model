
import ivm  from "isolated-vm";
import fs from "fs";

import {
Worker, isMainThread,parentPort,workerData
} from "node:worker_threads";
let chatInfer = null;
if(isMainThread){
    //import { chatInfer } from "./inference.js";
    chatInfer = (await import("./inference.js")).chatInfer;
}



async function runInIsolate(code){
    //return promise
    return new Promise(async (resolve, reject) => {
    setTimeout(() => {
        resolve("999");
        //reject("Error: Script took too long to execute.");
    }, 20000);
    
    let scriptOutputs = [];
    

    
    const isolate = new ivm.Isolate({ memoryLimit: 1024 });
    const context = isolate.createContextSync();
    const jail = context.global;
    jail.setSync('global', jail.derefInto());

    context.evalSync(`console = {};`);
    jail.setSync('log', function(...args) {
        scriptOutputs.push(JSON.stringify(args[0]));
        console.log("VM Message: ", ...args);
    });

    context.evalSync(`console.log = log;`);
    let err = false;
    try {
        let compiled = await isolate.compileScript(code);
        let evalOutput = await compiled.run(context, {promise: true});
        //check if not undefined or "", if not, push onto scriptOutputs
        if (evalOutput !== undefined && evalOutput !== "") {
            scriptOutputs.push(evalOutput);
        }
    }
    catch (e) {
        console.error(e);
        scriptOutputs.push("Error: " + e);
        err = true;
        resolve({text:scriptOutputs.join("\n"), image:null, error:err});
    }
    
    resolve({text:scriptOutputs.join("\n") ,error:err});
    

});

}


let testfile = fs.readFileSync("minimal.mjs",{encoding:"utf-8"});


let segments = testfile.split("//EVAL SECTION");

let evalsegment = segments[1];


async function test(codesegment){


    let result = await runInIsolate([codesegment,evalsegment].join("\n"));
    return parseFloat(result.text);
    let returnStr = "errors encountered: "+result.error+"\n\nfinal loss achieved after 5 seconds: "+result.text;

    return returnStr;
}

async function testMulti(codesegment){
    let workers = [];
    let promises = [];
    for(let i = 0; i < 4;i++){
        let worker = new Worker("./iterate.mjs");
        let prommy = new Promise((resolve,reject)=>{
            worker.on("message",msg=>{console.log(msg),resolve(msg);worker.terminate();});
            worker.on("online",()=>worker.postMessage(JSON.stringify({data:codesegment})))
        });
        promises.push(prommy);
    }
    await Promise.all(promises);
    let accum = 0;
    for(let i = 0; i < promises.length;i++){
        accum+=JSON.parse(await promises[i]).data;
    }
    accum = accum / promises.length;
    return accum;
}




let lastMessages = [];
if(isMainThread){


let currentBest = segments[0];
let iteration = 0;
let currentBestLoss = await testMulti(currentBest);
console.log("multi-test result: ",currentBestLoss);

while(true){
    try {
    iteration++;
    
    //let history = [{role:"user",content:"here is a pure JS implementation of a neural network and/or its optimizer. Optimizer and network are entirely separate; the optimizer only takes an arbitrary array of numbers (not neccessarily NN-specific) and an eval function. It should be noted that the goal of this whole project is backpropagation-free training using only forward passes and loss feedback. Optimize the architecture to more cleverly approximate gradients, leading to a lower loss at the same wall clock time. Implement a more aggressive initial exploratory phase. Experiment with only reassigning train_idx every few iterations. the output format of runLayer and runAll should unaffected. consider that the evaluate function is quite expensive. output the improved javascript code after [BEGIN JAVASCRIPT], after this tag nothing but pure javascript code should follow, without additional explanations. end the code with [END JAVASCRIPT]. Make sure all the class inputs and outptus/function signatures stay compatible, so the hidden benchmarking code still works. Afterwards the code will run for 10 seconds before the final loss is measured and returned to you. Before the begin tag, shortly explain what you plan on changing.\n\nhere is the code:\n\n"+currentBest}];
    let history = [{role:"user",content:"here is a pure JS implementation of a neural network and/or its optimizer. Optimizer and network are entirely separate; the optimizer only takes an arbitrary array of numbers (not neccessarily NN-specific) and an eval function. It should be noted that the goal of this whole project is backpropagation-free training using only forward passes and loss feedback. Optimize the architecture to more cleverly approximate gradients, leading to a lower loss at the same wall clock time. Implement a more aggressive initial exploratory phase. Experiment with only reassigning train_idx every few iterations. the output format of runLayer and runAll should unaffected. consider that the evaluate function is quite expensive. output the improved javascript code after [BEGIN JAVASCRIPT], after this tag nothing but pure javascript code should follow, without additional explanations. end the code with [END JAVASCRIPT]. Make sure all the class inputs and outptus/function signatures stay compatible, so the hidden benchmarking code still works.\n\nhere is the code:\n\n"+currentBest}];
    history = [...history, ...lastMessages]
    if(Math.random()>0.6){
        if(Math.random()>0.8){
            //history = [...history, ...lastMessages, {role:"user",content:"[HPARAM TUNING PHASE] change the hyperparameters instead of the architecture now."}]
        }
        
    }
    let think = false;
    if(Math.random()>0.9){
        //think=true;
    }
    let improvedVariations = await chatInfer(history, {n:1,max_tokens:32000,template_vars:{enable_thinking:think}});
    //for(let variation of improvedVariations){
    let variation=improvedVariations;

    let improved = variation.split("</think>").pop();
    let improvedText = improved;

    improved = improved.split("[BEGIN JAVASCRIPT]").pop();
    improved = improved.split("[END JAVASCRIPT]").shift();
    
    try {

        let modifiedLoss = await testMulti(improved);
        if(isNaN(modifiedLoss)){
            console.log("fuck, something NaN")
            continue;
        };
        if(modifiedLoss==0){
            console.log("hax");
            continue
        }
        console.log("baseline:",currentBestLoss,"proposed change:",modifiedLoss)
        if(modifiedLoss < currentBestLoss ){
            lastMessages.push({role:"assistant",content:improvedText});
            lastMessages.push({role:"user",content:"code executed! new loss: "+modifiedLoss});

            if(lastMessages.length > 8){
                lastMessages.shift();
                lastMessages.shift();
                lastMessages.shift();
                lastMessages.shift();
                lastMessages.shift();
            }

            fs.writeFileSync("./improvement"+iteration+".txt",improved);
            currentBest=improved;
            currentBestLoss=modifiedLoss;
        } else {
            lastMessages.push({role:"assistant",content:improvedText});
            lastMessages.push({role:"user",content:"code executed! new loss: "+modifiedLoss+"\nunfortunately, this is worse than before. try something else."});

        }
    } catch(e) {
        console.log(e);
    }
    //}
    } catch (e){
        console.log("this wouLdve been bad..",e);
    }
}
} else {
    parentPort.on("message",async msg=>{
        let res = await test(JSON.parse(msg).data)
        parentPort.postMessage(JSON.stringify({data:res}));
    });
}










