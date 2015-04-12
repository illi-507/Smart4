/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public Node outputNode=null;//the output node

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. The other nodes of inputNodes are of type
	 * input. The remaining nodes are of type sigmoid. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to output
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}

			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node
		outputNode=new Node(4);

		//Connecting hidden layer nodes with output node
		for(int i=0;i<hiddenNodes.size();i++)
		{
			NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(i),outputWeights[i]);
			outputNode.parents.add(nwp);
		}
	}

	/**
	 * Get the output from the neural network for a single instance
	 * 
	 * The parameter is a single instance
	 */

	public Double calculateOutputForInstance(Instance inst)
	{
		// TODO: add code here
		//set all the input node input values
		for(int i = 0; i < inst.attributes.size(); i++){
			inputNodes.get(i).setInput(inst.attributes.get(i));
		}

		//for each hidden node, calculate output, use those as inputs into output node
		//then calculate output value at output node
		for(int i = 0; i < hiddenNodes.size(); i++){
			hiddenNodes.get(i).calculateOutput();
		}

		outputNode.calculateOutput();
		return outputNode.getOutput();
	}





	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		double output;
		double teacher;
		double error;
		double value = 0;
		double adjustedWeight = 0;

		// TODO: add code here

		for(int i = 0; i < this.maxEpoch; i++){
			for(Instance e : trainingSet){
				output = calculateOutputForInstance(e);
				teacher = e.classValue;
				error = teacher - output;
				
				//backward pass
				//update hidden -> output
				for(int j = 0; j < outputNode.parents.size(); j++){
					NodeWeightPair hidden = outputNode.parents.get(j);
					adjustedWeight = this.learningRate * hidden.node.getOutput() * error * output *(1-output);
					value = hidden.weight * error * output * (1 - output);
					hidden.weight += adjustedWeight;

					//update input -> hidden
					if(hidden.node.parents != null){
						for(int k = 0; k < hidden.node.parents.size(); k++){
							NodeWeightPair inputNode = hidden.node.parents.get(k);
							inputNode.weight += this.learningRate * inputNode.node.getOutput() * hidden.node.getOutput() *
									(1 - hidden.node.getOutput()) * value;
						}
					}
					//value += adjustedWeight * error * output * (1 - output);
				}
			}
		}

	}
}