package error.backpropagation.algorithm;
/*
 @author Kunal Verma
*/
public interface NeuralNetInterface extends CommonInterface{
    final double bias = 1.0;
    public double sigmoid(double x);
    public double customSigmoid(double x);
    public void initialWeights();
    public void zeroWeights();
}
