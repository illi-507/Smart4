/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details
 * 
 * Do not modify. 
 * 
 * 
 */

import java.util.*;

public class Node{
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents=null; //Array List that will contain the parents (including the bias node) with weights if applicable
		 
	private Double inputValue=0.0;
	private Double outputValue=0.0;
	
	//Create a node with a specific type
	public Node(int type)
	{
		if(type>4 || type<0)
		{
			System.out.println("Incorrect value for node type");
			System.exit(1);
			
		}
		else
		{
			this.type=type;
		}
		
		if (type==2 || type==4)
		{
			parents=new ArrayList<NodeWeightPair>();
		}
	}
	
	//For an input node sets the input value which will be the value of a particular attribute
	public void setInput(Double inputValue)
	{
		if(type==0 || type == 2)//If input node
		{
			this.inputValue=inputValue;
			//System.out.println("Input values are: " + inputValue);
		}
	}

	/**
	 * Calculate the output of a sigmoid node.
	 * You can assume that outputs of the parent nodes have already been calculated
	 * You can get this value by using getOutput()
	 * @param train: the training set
	 */
	public void calculateOutput()
	{
		double inputTotal = 0.0;
		double outputTotal = 0.0;
		double parentValue, parentWeight;
		if(type==2)//Not an input or bias node
		{
			//for each parent get nodeweightpair and keep a total for inputs
			for(NodeWeightPair parent : parents){
				parentValue = parent.node.inputValue;
				parentWeight = parent.weight;
				inputTotal += parentValue * parentWeight;
			}
			
			//then calculate sigmoid function with that result
			outputTotal = 1 / (1 + Math.exp((-1)*inputTotal));
			this.outputValue = outputTotal;
		}
	}
	
	//Gets the output value
	public double getOutput()
	{
		
		if(type==0)//Input node
		{
			return inputValue;
		}
		else if(type==1 || type==3)//Bias node
		{
			return 1.00;
		}
		else
		{
			return outputValue;
		}
		
	}
}