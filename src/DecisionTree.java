import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * This class provides a framework for accessing a decision tree.
 * Put your code in constructor, printInfoGain(), buildTree and buildPrunedTree()
 * You can add your own help functions or variables in this class 
 */
public class DecisionTree {
	/**
	 * training data set, pruning data set and testing data set
	 */
	private DataSet train = null;		// Training Data Set
	private DataSet tune = null;		// Tuning Data Set
	private DataSet test = null;		// Testing Data Set
	private ArrayList<int[][]> data= null; //has counts of instances based on
	//the attribute value and its label
	private double[] infoGain; // info gain array for root node
	//root node
	DecTreeNode root = null;
	//best node in the tree to be pruned
	DecTreeNode bestPrune = null;
	/**
	 * Constructor
	 * 
	 * @param train  
	 * @param tune
	 * @param test
	 */
	DecisionTree(DataSet train, DataSet tune, DataSet test) {
		this.train = train;
		this.tune = tune;
		this.test = test;
		data = new ArrayList<int[][]>();
		infoGain = new double[train.attr_name.length];
	}

	//count the instances based on the attribute value and its label
	private void dataTable(List<Instance> trainList){
		for(Instance i : trainList){
			for(int j = 0; j < train.attr_name.length; j++){
				for(int k = 0; k < train.attr_val[j].length; k++){
					if(i.attributes.get(j).equals(train.attr_val[j][k])){
						if(i.label.equals(train.labels[0])){
							data.get(j)[0][k]++;
						}else{
							data.get(j)[1][k]++;
						}
					}
				}
			}
		}
	}


	/**
	 * print information gain of each possible question at root node.
	 * 
	 */
	public void printInfoGain()
	{

		boolean[] questions = new boolean[train.attr_name.length];
		//first calculate H(Y), which is the work without choosing
		//any attribute as root node
		double Hy = Hy(train.instances);
		infoGain = bestAttribute(Hy, train.instances, questions);
		String info = ": info gain = ";
		for(int i = 0; i < train.attr_name.length; i++){
			System.out.printf("%s%s%.3f\n", train.attr_name[i], info, infoGain[i]);
		}
	}

	//calculate the conditional entropy for each attribute and the information
	//gain for each attribute, return an array of info gain
	private double[] bestAttribute(double Hy, List<Instance> trainList, 
			boolean[] questions){
		data = new ArrayList<int[][]>();
		for(int i = 0; i < train.attr_name.length; i++){
			//first row represents 'e', second row represnets 'p'
			//every col represents each possible values for the attribute
			int[][] attri = new int[2][train.attr_val.length];
			data.add(attri);
		}
		dataTable(trainList);
		double[] info = new double[train.attr_name.length];
		for(int i = 0; i < train.attr_name.length; i++){
			//if the question has not been asked before by the parents
			if(!questions[i]){
				double[][] value = new double[train.attr_val[i].length][2];
				for(int j = 0; j < train.attr_val[i].length; j++){
					double sum = 0.0;
					double pE = 0.0;
					double pP = 0.0;
					sum = data.get(i)[0][j] + data.get(i)[1][j];
					if(sum != 0){
						pE = (double)data.get(i)[0][j]/sum;
						pP = (double)data.get(i)[1][j]/sum;
					}
					// if extreme case, namely either pE or pP is 0, 
					//the value H(Y|X=v) is 0;
					if(pE != 0 && pP != 0){
						value[j][0] = -1*(pE*log2(pE) + pP*log2(pP));
					}
					if(pP == 0 || pE == 0){
						value[j][0] = 0;
					}
					value[j][1] = sum/trainList.size();
					info[i] += value[j][0]*value[j][1];
				}
				info[i] = Hy - info[i];
			}
		}
		return info;
	}

	//calculate the H(Y) entropy for a particular list of instance
	//assume that their are only two classes as givin in the train DataSet
	private double Hy(List<Instance> ins){
		double Hy = 0.0;
		int countE = 0; //number of edible
		int countP = 0; //number of poisonous
		double size = ins.size();
		for(Instance i : ins){
			if(i.label.equals(train.labels[0])){
				countE++;
			}
			else{
				countP++;
			}
		}
		//if perfect homogeneity, entrorpy = 0
		if(countE == 0 || countP == 0){
			return 0;
		}
		double pE = countE/size;
		double pP = countP/size;
		Hy = -1*(pE*log2(pE) + pP*log2(pP));
		return Hy;
	}

	//select the best attribute based on the infoGain string
	private int selectAttr(double[] infoGain){
		double max = -1;
		int maxIndex = -1;
		for(int i = 0; i < infoGain.length; i++){
			if(infoGain[i] > max){
				maxIndex = i;
				max = infoGain[i];
			}
		}
		return maxIndex;
	}

	//calculate log base 2
	private double log2(double p){
		double result = 0.0;
		result = Math.log(p)/Math.log(2);
		return result;
	}

	//choose the list of instances based on the specific value of a 
	//specific attribute
	private List<Instance> updateEx(int index, String value, List<Instance> n){

		List<Instance> update = new ArrayList<Instance>();
		//add the instance with attribute attr of the value to the update List
		for(int i = 0; i < n.size(); i++){
			//is the attribute for this instance having this value?
			if(n.get(i).attributes.get(index).equals(value)){
				update.add(n.get(i));
			}
		}
		return update;
	}


	/**
	 * Build a decision tree given only a training set.
	 * 
	 */
	public void buildTree(){
		String defau = majorityVote(train.instances);
		boolean[] questions = new boolean[train.attr_name.length];
		root = new DecTreeNode(defau, null, "root", false);
		buildTree(root, train.instances, defau, questions);
	}


	//recursively build the tree
	private void buildTree(DecTreeNode node, List<Instance> ex, String defau, 
			boolean[] questions){
		boolean [] quest = questions.clone();
		String defa = defau;
		String label = majorityVote(ex);
		node.label = label;
		if(ex.size() == 0){
			//System.out.println("no example");
			node.label = defau;
			node.terminal = true;
			return;
		}
		//in the case of examples left all have the same labels
		if(sameLabel(ex)){
		//	System.out.println("have the same label");
			node.terminal = true;
			node.label = ex.get(0).label;
			return;
		}
		//in the case of empty attributes
		if(emptyAttr(questions)){
		//	System.out.println("run out of questions");
			node.terminal = true;
			return;
		}
		//best question to ask for the current node
		int bq = selectAttr(bestAttribute(Hy(ex), ex, questions));
		node.attribute = train.attr_name[bq];
		node.attriNum = bq;
		// n is the number of children
		int n = train.attr_val[bq].length;
		//have asked the bq question
		quest[bq] = true;
		defa = majorityVote(ex);
		for(int i = 0; i < n; i++){
			//parent attribute value
			String pAV = train.attr_val[bq][i];
			List<Instance> updated = updateEx(bq, train.attr_val[bq][i], ex);
			
			//choose based on the value of best attribute to form a new list of instance
			//System.out.println("Based on attribute " + train.attr_val[bq][i] + 
			//" num of items "+ t.size());
			DecTreeNode child = new DecTreeNode("", null, pAV, false);;
			//System.out.println(train.attr_val[bq][i]);
			//System.out.println("we are asking the question " + 
			//node.attribute + " Based on attribute " 
			//+ train.attr_val[bq][i] + " num of items "+updateEx(bq, 
			//train.attr_val[bq][i], ex).size());
			buildTree(child, updated, defa, quest);
			node.addChild(child);
		}
	}

	private String majorityVote(List<Instance> ex){
		//initialize the majority as the first label of the train DataSet
		String label = train.labels[0];
		int numE = 0;
		int numP = 0;
		for(int i = 0; i < ex.size(); i++){
			if(ex.get(i).label.equals(train.labels[0])){
				numE++;
			}else{
				numP++;
			}
		}

		if(numE < numP){
			return train.labels[1];
		}
		//in the case of tie in majority vote, return the first one listed
		//in the train set
		return label;
	}

	//based on the questions boolean array that is passed, decide if there
	//is any candidate questions that are left
	private boolean emptyAttr(boolean[] questions){
		for(int i = 0; i < questions.length; i++){
			if(!questions[i]){
				return false;
			}
		}
		return true;
	}

	//check if the training List of Instances all have the same label 
	private boolean sameLabel(List<Instance> ex){
		if(ex.size() == 0){
			return false;
		}
		String label = ex.get(0).label;
		for(Instance i : ex){
			if(!i.label.equals(label)){
				return false;
			}
		}
		return true;
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 */
	public void buildPrunedTree() {
		//indicate if we stop pruning
		boolean stop = false;
		//firstly build the tree
		buildTree();
		//calculate the accurcy of tuning set before pruning
		double accuracy = calcTestAccuracy(tune, classifyTune());
		while(!stop){
			//pass in root node to find the best prune node
			buildPrunedTree(root, -1);
			//save the best prune node's children
			List<DecTreeNode> temp = bestPrune.children;
			//prune the best prune node
			bestPrune.children = null;
			bestPrune.terminal = true;
			//calculate the after prune tuning set accuracy
			double pruneAcu = calcTestAccuracy(tune, classifyTune());
			//if we find that the accuracy decreased
			if(pruneAcu < accuracy){
				//stop pruning
				stop = true;
				//recover the pruned node
				bestPrune.children = temp;
				bestPrune.terminal = false;
				//if the tuning set accuracy does not decrease
				//update the prune accuracy
			}else{
				accuracy = pruneAcu;
			}
		}
	}

	//do DFS to the whole tree for one iteration
	private void buildPrunedTree(DecTreeNode prune, double best){
		//DecTreeNode bestPruneNode = null;
		if(prune.terminal){
			return;
		}
		//visit the current node
		//if the after pruned accuracy is strictly larger than 
		//the so-far best accuracy, update the so-far best accuracy
		//and update the best prune node
		if(pruneSingleNode(prune) > best){
			best = pruneSingleNode(prune);
			bestPrune = prune;
		}
		//recursively visit current node's children
		for(int i = 0; i < prune.children.size(); i++){
			buildPrunedTree(prune.children.get(i), best);
		}
		return;
	}

	// prune the single node that is passed in, return the tuning set accuracy after 
	// pruning the node
	private double pruneSingleNode(DecTreeNode prune){
		//initialize tuning set accuracy to 0
		double accuracy = 0.0;
		List<DecTreeNode> temp;
		//save the pruned node's children
		temp = prune.children;
		//prune the node by setting its children to null
		//and set terminal = true
		prune.children = null;
		prune.terminal = true;
		//calculate tuning set accuracy
		accuracy = calcTestAccuracy(tune, classifyTune());
		//revert back the pruned node
		prune.children = temp;
		prune.terminal = false;
		return accuracy;
	}

	//return an array of predication labels for tuning set
	private String[] classifyTune(){
		String[] results = new String[tune.instances.size()];
		for(int i = 0; i < tune.instances.size(); i++){
			results[i] = classify(root, tune.instances.get(i));
		}
		return results;
	}
	//calculate the accuracy for a dataset based on an array of prediction
	private static double calcTestAccuracy(DataSet test, String[] results) {
		List<Instance> testInsList = test.instances;
		int correct = 0, total = testInsList.size();
		for(int i = 0; i < testInsList.size(); i ++){
			if(testInsList.get(i).label.equals(results[i])){
				correct++;
			}
		}
		return correct * 1.0 / total;
	}

	/**
	 * Evaluates the learned decision tree on a test set.
	 * @return the label predictions for each test instance 
	 * 	according to the order in data set list
	 */
	public String[] classify() {
		String[] results = new String[test.instances.size()];
		for(int i = 0; i < test.instances.size(); i++){
			results[i] = classify(root, test.instances.get(i));
		}
		return results;
	}
	//return the prediction label of the instance using the decision tree
	private String classify(DecTreeNode node, Instance test){
		String result = "";
		if (node.terminal) {
			result = node.label;
		}else{
			String attri = test.attributes.get(node.attriNum);
			for(int i = 0; i < node.children.size(); i++){
				if(node.children.get(i).parentAttributeValue.equals(attri)){
					node = node.children.get(i);
					break;
				}
			}
			result = classify(node, test);
		}
		return result;
	}

	/**
	 * Prints the tree in specified format. It is recommended, but not
	 * necessary, that you use the print method of DecTreeNode.
	 * 
	 * Example:
	 * Root {odor?}
	 * ?
	 *     a (e)
	 *     m (e)
	 *	   n {habitat?}
	 *         g (e)
	 *  	   l (e)
	 *	   p (p)
	 *	   s (e)
	 *         
	 */
	public void print() {
		root.print(0);
	}
}
