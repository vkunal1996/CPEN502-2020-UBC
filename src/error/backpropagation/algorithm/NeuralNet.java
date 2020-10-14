package error.backpropagation.algorithm;
/*
    @author: vkunal1996@gmail.com
* */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface{
    /*
        Public member of the Class.
    */
    public static int numberOfInputs; /*Number of input =3 with one input value as a bias*/
    public static int numberOfNeurons;/*Number of Neurons=5 with one bias neuron*/
    public static int numberOfOutput; /*Number of output =1 for each input triplet*/
    public static double learningRate; /*Learning rate of the algorithm*/
    public static double momentumTerm;/*Momentum in the training*/

    public static double[][] weightsInputtoHidden;/*Weights assigned from Input to Hidden Layer*/
    public static double[] weightsHiddenToOutputs;/*Weights assigned from Hidden to Output Layer*/
    public static double[][] binaryXORInput ={
            { 1.0, 0.0, 1.0 },
            { 0.0, 1.0, 1.0 },                                      /*Input Vector Binary (0,1)*/
            { 1.0, 1.0, 1.0 },
            { 0.0, 0.0, 1.0 }
    };
    public static double[][] bipolarXORInput = {
            { -1.0, -1.0, 1.0 },
            { -1.0, 1.0, 1.0 },                                    /*Input Vector Bipolar (1,-1)*/
            { 1.0,-1.0, 1.0 },
            { 1.0, 1.0, 1.0 }
    };
    public static double[] binaryXOROutput = {
            1.0,
            1.0,                                                    /*Output Vector binary*/
            0.0,
            0.0
    };
    public static double[] bipolarXOROutput = {
            -1.0,
            1.0,                                                    /*Output Vector Bipolar*/
            1.0,
            -1.0
    };
    public static double[][] inputArray; /*general Variable, to chose whether the input is binary or bipolar */
    public static double[] output;       /*general Variable, to chose whether the outputvector is binary or bipolar*/
    public static double lower;          /*Lower Bound for Activation Function*/
    public static double upper;          /*Upper Bound for Activation Function*/
    public static double[] outputComing ;/*OutputComing from the Layer*/
    public static double[] weightChangeForOutput; /*Storage for weight change in output layer*/
    public static double[][] weightChangeForHidden; /*Storage for weight change in Hidden Layer*/
    public static double[] numberOfHiddenNeurons; /*General Variable for number of neurons for the assignment*/

    public NeuralNet(int inputs, int hiddenInputs, int output, double learningRate, double momentumTerm,boolean isBinary) {
        /*Initialising the required Values*/
        this.numberOfInputs = inputs;
        this.numberOfNeurons = hiddenInputs;
        this.numberOfOutput = output;
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;
        this.weightsInputtoHidden = new double[numberOfInputs][numberOfNeurons];
        this.weightsHiddenToOutputs = new double[numberOfNeurons];
        this.weightChangeForOutput = new double[numberOfNeurons];
        this.weightChangeForHidden = new double[numberOfInputs][numberOfNeurons];
        this.numberOfHiddenNeurons = new double[numberOfNeurons];

        /*Checking if the type of input is binary or not*/
        if(isBinary){
            this.inputArray=binaryXORInput;
            this.output=binaryXOROutput;
            this.lower=0.0;
        }
        else{
            this.inputArray=bipolarXORInput;
            this.output=bipolarXOROutput;
            this.lower=-1.0;
        }
        this.upper=1.0;
        this.outputComing=new double[this.inputArray.length];
    }

    @Override
    public double sigmoid(double x) {
        return 1/(1+Math.pow(Math.E,-x)); /*Signmoid Activation Function*/
    }

    @Override
    public double customSigmoid(double x) {
        return (upper - lower) / (1 + Math.pow(Math.E, -x)) + lower; /*Calculation of custom Sigmoid Value*/
    }

    @Override
    public void initialWeights() {
        for (int i = 0; i < numberOfInputs; i++) {
            for (int j = 0; j < numberOfNeurons - 1; j++) {
                weightsInputtoHidden[i][j] = (Math.random() - 0.5);
            }                                                           /*Initialising the required weights.*/
        }
        for (int i = 0; i < numberOfNeurons; i++) {
            weightsHiddenToOutputs[i] = (Math.random() - 0.5);
        }
    }



    @Override
    public void zeroWeights() {
        // to do
        for (int i = 0; i < numberOfInputs; i++) {
            for (int j = 0; j < numberOfNeurons - 1; j++) {
                weightsInputtoHidden[i][j] = 0.0;
            }                                           /*Assigning the Zero Weights*/
        }
        for (int i = 0; i < numberOfNeurons; i++) {
            weightsHiddenToOutputs[i] = 0.0;
        }
    }

    public double feedForward(double[] inputArray, double[][] weightsinp,double[] hiddenArray, double[] weightsout){
        for (int i = 0; i < numberOfNeurons - 1; i++) {
            numberOfHiddenNeurons[i] = 0.0;
            for (int j = 0; j < numberOfInputs; j++) {                          /*feed forward from input to hidden*/
                numberOfHiddenNeurons[i] += inputArray[j] * weightsinp[j][i];
            }
            numberOfHiddenNeurons[i] = customSigmoid(numberOfHiddenNeurons[i]);
        }                                                                      /*Feedforward Propagation*/
        numberOfHiddenNeurons[numberOfNeurons - 1] = bias;
        double result = 0.0;
        for (int i = 0; i < numberOfNeurons; i++) {
            result += hiddenArray[i] * weightsout[i];       /*from hidden to output*/
        }
        return result;

    }
//

    public void forwardPropagation(int i) {
        outputComing[i]=feedForward(inputArray[i],weightsInputtoHidden,numberOfHiddenNeurons,weightsHiddenToOutputs);
        outputComing[i] = customSigmoid(outputComing[i]); /*Calculating custom sigmoid*/

    }

    double Error() {
        double totalError = 0.0;
        for (int i = 0; i < output.length; i++) {
            double singleError = Math.pow(outputComing[i] - output[i], 2);
            totalError += singleError;          /*Calculating the Error for each epoch and then finding the total associated error*/
        }
        return totalError / 2;
    }



    public double customSigmoidDerivative(double y) {
        double result;
        result = (1.0 / (upper - lower)) * (y - lower) * (upper - y);
        return result;                  /*Custom Sigmoid derivative: Referenced from Laurene Fausett.*/
    }

    double deltaOutput;
    double deltaForOutput(double outputComing, double output) {
        return customSigmoidDerivative(outputComing) * (output - outputComing); /*Updating the delta values*/
    }

    void backPropagation(int batch) {
        deltaOutput = deltaForOutput(outputComing[batch], output[batch]); // delta for output calculation
        for (int i = 0; i < numberOfNeurons; i++) {
            weightChangeForOutput[i] = learningRate * deltaOutput * numberOfHiddenNeurons[i] + momentumTerm * weightChangeForOutput[i]; // calculating weight change
        }

        /*Updating the weights for outputs*/
        for (int i = 0; i < numberOfNeurons; i++) {
            weightsHiddenToOutputs[i] += weightChangeForOutput[i];
        }

        /* Calcualation of delta*/
        for (int i = 0; i < numberOfNeurons - 1; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                double delta = 0.0;
                delta = customSigmoidDerivative(numberOfHiddenNeurons[i]) * weightsHiddenToOutputs[i] * deltaOutput;
                weightChangeForHidden[j][i] = learningRate * inputArray[batch][j] * delta + momentumTerm * weightChangeForHidden[j][i];
                weightsInputtoHidden[j][i] += weightChangeForHidden[j][i];
            }
        }

    }

    @Override
    public double outputFor(double[] X) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public void save(File argFile) {
        // TODO Auto-generated method stub

    }

    @Override
    public void load(String argFileName) throws IOException {
        // TODO Auto-generated method stub

    }

    public void train(String fileName) {
        /*Training the Neural Network*/
        double error = 1;
        String errorRecord="";
        int epoch = 0;
        while (error > 0.05) {
            for (int i = 0; i < inputArray.length; i++) {
                forwardPropagation(i);
                backPropagation(i);
            }
            error = Error();
            errorRecord+=error+",";

            epoch++;
        }
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(fileName, true));
            out.write(epoch+","+errorRecord+"\n");
            out.close();
        } catch (IOException e) {
            System.out.println("exception occoured" + e);
        }
    }

    public static void main(String[] args) {
        /*Object Creation*/
        NeuralNet XOR = new NeuralNet(3, 5, 1, 0.2, 0.9,false);
       /*File for saving the errors*/
        String filename="bipolar_momentum_XOR.csv";

        int n=100;
        /*Number of Runs*/
        for(int i=0;i<n;i++) {
            XOR.initialWeights();
            XOR.train(filename);
            System.out.println("Completed "+(i+1));
        }
        System.out.println("Results Stored in the file named: "+filename);
    }
}
