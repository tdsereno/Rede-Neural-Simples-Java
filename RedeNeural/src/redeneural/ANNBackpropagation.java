package redeneural;

import java.util.Random;

/**
 * Artificial Neural Network with Backpropagation Training Algorithm.
 *
 * adaptado de https://github.com/ard333/ANN-Backpropagation
 */
public final class ANNBackpropagation {

    private final Integer numOfInput;
    private final Integer numOfHidden;
    private final Integer numOfOutput;

    private final Double learningRate;


    private Double[] X;//entrada
    private Double[] Y;//intermediaria
    private Double[] Z;//saida

    private Double[][] w1;//pesos da camada entrada->intermediaria
    private Double[][] w2;//pessos da camada intermediaria->saida

    private Double[] sigmaForY;
    private Double[] sigmaForZ;

    private Double[][] deltaw1;
    private Double[][] deltaw2;

    private Double[][] inputTraining;
    private Double[][] expectedOutput;

    private Integer epoch;
    private ActivationFunction activationFunction;

    /**
     * Create new Artificial Neural Network with specify the number of neurons,
     * learning rate and minimum error.
     *
     * @param numOfInput number of input unit.
     * @param numOfHidden number of hidden neuron.
     * @param numOfOutput number of output neuron.
     * @param learningRate learning rate (0.1 - 1).
    
     */
    public ANNBackpropagation(
            Integer numOfInput, Integer numOfHidden, Integer numOfOutput,
            Double learningRate,  ActivationFunction activationFunction
    ) {
        this.numOfInput = numOfInput;
        this.numOfHidden = numOfHidden;
        this.numOfOutput = numOfOutput;
        this.learningRate = learningRate;
     
        this.activationFunction = activationFunction;

        this.init();
    }

    /**
     * Inicializa os arrays e define pesos aleatorios
     */
    private void init() {

        this.epoch = 0;

        this.X = new Double[numOfInput + 1];
        this.Y = new Double[numOfHidden + 1];
        this.Z = new Double[numOfOutput];
        this.X[numOfInput] = 1.0;//bias at last index
        this.Y[numOfHidden] = 1.0;//bias at last index

        this.sigmaForY = new Double[numOfHidden + 1];
        this.sigmaForZ = new Double[numOfOutput];

        this.w1 = new Double[numOfInput + 1][numOfHidden]; 
        this.w2 = new Double[numOfHidden + 1][numOfOutput];
        this.deltaw1 = new Double[numOfInput + 1][numOfHidden];
        this.deltaw2 = new Double[numOfHidden + 1][numOfOutput];

        Random r = new Random();

        for (int i = 0; i < this.numOfInput + 1; i++) { // percorre todos neuronios da camada da entrada
            for (int j = 0; j < this.numOfHidden; j++) { //percorre todos neuronios da camada intermediaria
                this.w1[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1 define pesosa aleatorios para a camada de entrada -> intermediaria
            }
        }
        for (int i = 0; i < numOfHidden + 1; i++) { // percorre todos neuronios da camada intermediaria 
            for (int j = 0; j < numOfOutput; j++) { // percorre todos neuronios da camada de saida
                this.w2[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1 define pesos aleatorios para a camada intermediaria -> saida
            }
        }
    }

    /**
     * Seta os dados de treinamento.
     *
     * @param inputTraining seta os dados para treinamento.
     * @param expectedOutput seta o resultado esperado.
     */
    public void setTrainingData(Double[][] inputTraining, Double[][] expectedOutput) {
        this.inputTraining = inputTraining;
        this.expectedOutput = expectedOutput;
    }

    /**
     * Treina ate que a taxa com os dados de entrada times vezes
     */
    public Double train(int times) {
        Double[] eO = new Double[numOfOutput];
        Double erro = 0.0;
        if (this.inputTraining != null && this.expectedOutput != null) {
            System.out.println("Treinando, por favor aguarde");
            Double err = 0.0;
            while (times > 0) {
                this.epoch++;
                for (int i = 0; i < this.inputTraining.length; i++) {
                    System.arraycopy(this.inputTraining[i], 0, X, 0, this.inputTraining[i].length); // copiamos os dados de entrada para X
                    System.arraycopy(this.expectedOutput[i], 0, eO, 0, this.expectedOutput[i].length); //copiamos os dados de resultado esperado para eO

                    this.feedForward();
                    this.backPropagation(eO);
                }
                err = this.caclERR();
                System.out.println("Taxa de erro: " + err);
                times--;
                erro = err;
            } 
        } else {
            System.out.println("Sem dados para treinar");
        }
        return erro;
    }

    /**
     * Calculate error average for all pattern.
     *
     * @return error average.
     */
    private Double caclERR() {
        Double[] eO = new Double[numOfOutput];
        Double err = 0.0;
        Double errTotal = 0.0;

        for (int i = 0; i < this.inputTraining.length; i++) {
            System.arraycopy(this.inputTraining[i], 0, X, 0, this.inputTraining[i].length);
            System.arraycopy(this.expectedOutput[i], 0, eO, 0, this.expectedOutput[i].length);
            this.feedForward();
            for (int a = 0; a < this.numOfOutput; a++) {
                err += Math.pow((eO[a] - this.Z[a]), 2);
            }
            err /= numOfOutput;
            errTotal += err;
        }
        errTotal /= this.inputTraining.length;
        return errTotal;
    }

    /**
     * Test pattern after training.
     *
     * @param input input pattern.
     */
    public void test(Double[] input) {
        System.arraycopy(input, 0, this.X, 0, this.numOfInput);
        this.feedForward();
    }

    /**
     * Feed-forward.
     */
    private void feedForward() {
        this.setOutputY();
        this.setOutputZ();
    }

    /**
     * Calculate each output of hidden neuron.
     */
    private void setOutputY() {
        for (int a = 0; a < numOfHidden; a++) {
            this.sigmaForY[a] = 0.0;
        }
        for (int j = 0; j < this.numOfHidden; j++) {
            for (int i = 0; i < this.numOfInput + 1; i++) {
                try {
                    this.sigmaForY[j] = this.sigmaForY[j] + this.X[i] * this.w1[i][j];
                } catch (Exception e) {
                    System.out.println("erro" + e);
                }

            }
        }
        for (int j = 0; j < numOfHidden; j++) {
            if (null != this.activationFunction) {
                switch (this.activationFunction) {
                    case SIGMOID:
                        this.Y[j] = this.sigmoid(this.sigmaForY[j]);
                        break;
                    case BIPOLAR_SIGMOID:
                        this.Y[j] = this.bipolarSigmoid(this.sigmaForY[j]);
                        break;
                    case TANH:
                        this.Y[j] = this.tanH(this.sigmaForY[j]);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    /**
     * Calculate each output of output neuron.
     */
    private void setOutputZ() {
        for (int a = 0; a < numOfOutput; a++) {
            this.sigmaForZ[a] = 0.0;
        }
        for (int k = 0; k < this.numOfOutput; k++) {
            for (int j = 0; j < this.numOfHidden + 1; j++) {
                this.sigmaForZ[k] = this.sigmaForZ[k] + this.Y[j] * this.w2[j][k];
            }
        }
        for (int k = 0; k < this.numOfOutput; k++) {
            if (null != this.activationFunction) {
                switch (this.activationFunction) {
                    case SIGMOID:
                        this.Z[k] = this.sigmoid(this.sigmaForZ[k]);
                        break;
                    case BIPOLAR_SIGMOID:
                        this.Z[k] = this.bipolarSigmoid(this.sigmaForZ[k]);
                        break;
                    case TANH:
                        this.Z[k] = this.tanH(this.sigmaForZ[k]);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    /**
     * Backpropagation.
     *
     * @param expectedOutput set of expected output.
     */
    private void backPropagation(Double[] expectedOutput) {
        Double[] fO = new Double[this.numOfOutput];

        for (int k = 0; k < numOfOutput; k++) {
            if (null != this.activationFunction) {
                switch (this.activationFunction) {
                    case SIGMOID:
                        fO[k] = (expectedOutput[k] - this.Z[k]) * this.sigmoidDerivative(this.sigmaForZ[k]);
                        break;
                    case BIPOLAR_SIGMOID:
                        fO[k] = (expectedOutput[k] - this.Z[k]) * this.bipolarSigmoidDerivative(this.sigmaForZ[k]);
                        break;
                    case TANH:
                        fO[k] = (expectedOutput[k] - this.Z[k]) * this.tanHDerivative(this.sigmaForZ[k]);
                        break;
                    default:
                        break;
                }
            }
        }
        for (int j = 0; j < this.numOfHidden + 1; j++) {//+bias weight
            for (int k = 0; k < this.numOfOutput; k++) {
                this.deltaw2[j][k] = this.learningRate * fO[k] * this.Y[j];
            }
        }
        Double[] fHNet = new Double[this.numOfHidden];
        for (int j = 0; j < this.numOfHidden; j++) {
            fHNet[j] = 0.0;
            for (int k = 0; k < this.numOfOutput; k++) {
                fHNet[j] = fHNet[j] + (fO[k] * this.w2[j][k]);
            }
        }
        Double[] fH = new Double[this.numOfHidden];
        for (int j = 0; j < this.numOfHidden; j++) {
            if (null != this.activationFunction) {
                switch (this.activationFunction) {
                    case SIGMOID:
                        fH[j] = fHNet[j] * this.sigmoidDerivative(this.sigmaForY[j]);
                        break;
                    case BIPOLAR_SIGMOID:
                        fH[j] = fHNet[j] * this.bipolarSigmoidDerivative(this.sigmaForY[j]);
                        break;
                    case TANH:
                        fH[j] = fHNet[j] * this.tanHDerivative(this.sigmaForY[j]);
                        break;
                    default:
                        break;
                }
            }
        }
        for (int i = 0; i < this.numOfInput + 1; i++) {
            for (int j = 0; j < numOfHidden; j++) {
                this.deltaw1[i][j] = this.learningRate * fH[j] * this.X[i];
            }
        }
        this.changeWeight();
    }

    /**
     * Atualiza todos pesos
     */
    private void changeWeight() {
        for (int j = 0; j < numOfHidden + 1; j++) {
            for (int k = 0; k < numOfOutput; k++) {
                this.w2[j][k] = this.w2[j][k] + this.deltaw2[j][k];
            }
        }
        for (int i = 0; i < numOfInput + 1; i++) {
            for (int j = 0; j < numOfHidden; j++) {
                this.w1[i][j] = this.w1[i][j] + this.deltaw1[i][j];
            }
        }
    }

    /**
     * Sigmoid Activation Function.
     * <br/>f(x) = 1 / (1 + exp(-x))
     *
     * @param x an input value.
     * @return a result of Sigmoid Activation Function.
     */
    private Double sigmoid(Double x) {
        return 1 / (1 + (double) Math.exp(-x));
    }

    /**
     * Derivative of Sigmoid Activation Function.
     * <br/>f'(x) = f(x) * (1 - f(x))
     *
     * @param x an input value.
     * @return a result of Derivative Sigmoid Activation Function.
     */
    private Double sigmoidDerivative(Double x) {
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }

    /**
     * Sigmoid Bipolar Activation Function.
     * <br/>f(x) = 2 / (1 + exp(-x)) - 1
     *
     * @param x an input value.
     * @return a result of Sigmoid Bipolar Activation Function.
     */
    private Double bipolarSigmoid(Double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    /**
     * Derivative of Sigmoid Bipolar Activation Function.
     * <br/>f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
     *
     * @param x an input value.
     * @return a result of Derivative Sigmoid Bipolar Activation Function.
     */
    private Double bipolarSigmoidDerivative(Double x) {
        return 0.5 * (1 + this.bipolarSigmoid(x)) * (1 - this.bipolarSigmoid(x));
    }

    /**
     * TanH Activation Function.
     * <br/>f(x) = 2 / (1 + exp(-x)) - 1
     * <br/>output range -1 until 1.
     *
     * @param x an input value.
     * @return a result of TanH Activation Function.
     */
    private Double tanH(Double x) {
        return 2 / (1 + Math.exp(-2 * x)) - 1;
    }

    /**
     * Derivative of TanH Activation Function.
     * <br/>f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
     * <br/>output range -1 until 1.
     *
     * @param x an input value.
     * @return a result of Derivative TanH Activation Function.
     */
    private Double tanHDerivative(Double x) {
        return 1 - Math.pow(this.tanH(x), 2);
    }

    /**
     * Method for getting output of each output neuron.
     *
     * @return output of each output neuron.
     */
    public Double[] getOutput() {
        return this.Z;
    }

    /**
     * Method for getting epoch until minimum error reached.
     *
     * @return epoch until minimum error reached.
     */
    public Integer getEpoch() {
        return this.epoch;
    }

}
