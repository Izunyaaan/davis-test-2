import { MnistData, IMAGE_H, IMAGE_W } from './data.js';

function createConvModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'
  }));
  return model;
}

function createConvDropModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.169 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'
  }));
  return model;
}


function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [IMAGE_H, IMAGE_W, 1] }));
  model.add(tf.layers.dense({ units: 200, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  return model;
}

async function train(model, onIteration, inputElement, outputElement) {
  inputElement.disabled = false;
  const optimizer = "SGD"
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const batchSize = 320;
  const validationSplit = 0.15;
  const trainEpochs = 3;
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
    Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
    trainEpochs;

  let valAcc;
  let trainAcc
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        console.log(`Training... (` +
          `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
          ` complete). To stop training, refresh or close page.`);

        progress.testVar = `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%`
        inputElement.innerText = "Stop"
        outputElement.innerText = "Training... Progress " + progress.testVar



        if (stopTraining) {
          model.stopTraining = true;
        }
        //
        if (onIteration && batch % 10 === 0) {
          onIteration('onBatchEnd', batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        trainAcc = logs.acc

        if (stopTraining) {
          model.stopTraining = true;
        }
        inputElement.innerText = "Done. Re-train the model?"

        //
        if (onIteration) {
          onIteration('onEpochEnd', epoch, logs);
        }
        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  const finalTrainAccPercent = trainAcc * 100;

  isTraining = !isTraining
  outputElement.innerText = `Final Training Accuracy: ${finalTrainAccPercent.toFixed(1)}%
  Final validation accuracy: ${finalValAccPercent.toFixed(1)}%
  Final test accuracy: ${testAccPercent.toFixed(1)}%`
  console.log(
    `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
    `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
  disableButtons(false)
}

async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  tf.tidy(() => {
    const output = model.predict(examples.xs);
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

  });
}




let data;

async function loadANN() {
  data = new MnistData();
  await data.load();
  const model = createDenseModel();
  model.summary();

  await train(model, () => showPredictions(model), trainDenseModelBtn, trainDenseModelLabel);
}

async function loadCNN() {
  data = new MnistData();
  await data.load();
  const model = createConvModel();
  model.summary();
  await train(model, () => showPredictions(model), trainConvModelBtn, trainConvModelLabel);

}
async function loadCNNwDO() {
  data = new MnistData();
  await data.load();
  const model = createConvDropModel();
  model.summary();
  await train(model, () => showPredictions(model), trainConvDropModelBtn, trainConvDropModelLabel);

}

const trainConvModelBtn = document.getElementById('convModelBtn')
const trainConvModelLabel = document.getElementById('convProgress')
const trainDenseModelBtn = document.getElementById('denseModelBtn')
const trainDenseModelLabel = document.getElementById('denseProgress')
const trainConvDropModelBtn = document.getElementById('convDropModelBtn')
const trainConvDropModelLabel = document.getElementById('convDropProgress')
let stopTraining = false
let isTraining = false
let progress = {
  value: '',
  get testVar() {
    return this.value;
  },
  set testVar(value) {
    this.value = value;
  }
}

const disableButtons = (isDisabled) => {
  trainConvModelBtn.disabled = isDisabled
  trainDenseModelBtn.disabled = isDisabled
  trainConvDropModelBtn.disabled = isDisabled
}

const trainConvModel = () => {
  if (!isTraining) {
    isTraining = !isTraining
    stopTraining = false
    trainConvModelBtn.innerText = "Starting..."
    disableButtons(true)
    loadCNN()
  } else {
    isTraining = !isTraining
    stopTraining = true
    disableButtons(false)
  }
}

const trainDenseModel = () => {
  if (!isTraining) {
    isTraining = !isTraining
    stopTraining = false
    trainDenseModelBtn.innerText = "Starting"
    disableButtons(true)
    loadANN()
  } else {
    isTraining = !isTraining
    stopTraining = true
    disableButtons(false)
  }
}
const trainConvDropModel = () => {
  if (!isTraining) {
    isTraining = !isTraining
    stopTraining = false
    trainConvDropModelBtn.innerText = "Starting"
    disableButtons(true)
    loadCNNwDO()
  } else {
    isTraining = !isTraining
    stopTraining = true
    disableButtons(false)
  }
}

trainConvModelBtn.addEventListener("click", trainConvModel)
trainDenseModelBtn.addEventListener("click", trainDenseModel)
trainConvDropModelBtn.addEventListener("click", trainConvDropModel)
