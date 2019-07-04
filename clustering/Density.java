package fansactive.clustering;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import weka.core.Instances;

public class Density extends Instances {
	/**
	 * The total run time
	 */
	public static long totalRuntime;

	/**
	 * The distance measure.
	 */
	public static final int EUCLIDIAN = 0;

	/**
	 * The distance measure.
	 */
	public static final int MANHATTAN = 1;

	/**
	 * The distance measure.
	 */
	int distanceMeasure;

	/**
	 * Data type.
	 */
	public static final int INFORMATION_SYSTEM = 0;

	/**
	 * Data type.
	 */
	public static final int DECISION_SYSTEM = 1;

	/**
	 * Data type.
	 */
	int dataType;

	/**
	 * How many labels are taught.
	 */
	int numTeach;

	/**
	 * How many labels are predicted.
	 */
	int numPredict;

	/**
	 * How many labels are voted.
	 */
	int numVote;

	/**
	 * dc
	 */
	double dc;

	/**
	 * RhoIndex
	 */
	int[] ordrho;

	/**
	 * Rho
	 */
	double[] rho;

	/**
	 * Delta
	 */
	double[] delta;

	/**
	 * Priority
	 */
	double[] priority;

	/**
	 * The node index of centers
	 */
	int[] centers;

	/**
	 * The maximal distance between any pair of points.
	 */
	double maximalDistance;

	/**
	 * The maximal delta
	 */
	double maximalDelta;

	/**
	 * The cluster information. Which cluster do I belong to?
	 */
	int[] clusterIndices;

	/**
	 * The block information.
	 */
	int[][] blockInformation;

	/**
	 * Who is my master?
	 */
	int[] master;

	/**
	 * Predicted labels.
	 */
	int[] predictedLabels;

	/**
	 * Is respective instance already classified? If so, we do not process it
	 * further.
	 */
	boolean[] alreadyClassified;

	/**
	 * The descendant indices to show the importance of instances in a
	 * descendant order.
	 */
	int[] descendantIndices;

	/**
	 * How many instances are taught for each block.
	 */
	int teachInstancesForEachBlock;

	/**
	 ********************************** 
	 * Read from a reader
	 ********************************** 
	 */
	public Density(Reader paraReader) throws IOException, Exception {
		super(paraReader);
		initialize();
	}// Of the first constructor

	/**
	 *************** 
	 * Get the numTeach.
	 **************** 
	 */
	public double getnumTeach() {
		return numTeach;
	}// Of getRi

	/**
	 ********************************** 
	 * Initialize.
	 ********************************** 
	 */
	public void initialize() {
		setDistanceMeasure(MANHATTAN);
		setDataType(DECISION_SYSTEM);
		computeMaximalDistance();
	}// Of initialize

	/**
	 ********************************** 
	 * Set Dc from maximal distance.
	 ********************************** 
	 */
	public void setDcFromMaximalDistance(double paraPercentage) {
		dc = maximalDistance * paraPercentage;
		// System.out.println("dc " + dc);
	}// Of setDcFromMaximalDistance

	/**
	 ********************************** 
	 * Compute distance between instances.
	 ********************************** 
	 */
	public double distance(int paraI, int paraJ, int paraMeasure) {
		distanceMeasure = paraMeasure;
		return distance(paraI, paraJ);
	}// Of distance

	/**
	 ********************************** 
	 * Set the distance measure.
	 ********************************** 
	 */
	public void setDistanceMeasure(int paraMeasure) {
		distanceMeasure = paraMeasure;
	}// Of setDistanceMeasure

	/**
	 ********************************** 
	 * Set the data type.
	 ********************************** 
	 */
	public void setDataType(int paraDataType) {
		dataType = paraDataType;
		setClassIndex(numAttributes() - 1);
	}// Of setDataType

	/**
	 ********************************** 
	 * Compute the maximal distance.
	 ********************************** 
	 */
	public double computeMaximalDistance() {
		maximalDistance = 0;
		double tempDistance;
		for (int i = 0; i < numInstances(); i++) {
			for (int j = 0; j < numInstances(); j++) {
				tempDistance = distanceManhattan(i, j);
				if (maximalDistance < tempDistance) {
					maximalDistance = tempDistance;
				}// Of if
			}// Of for j
		}// Of for i
			// System.out.println("maxdistance are " + maximalDistance);
		return maximalDistance;
	}// Of setDistanceMeasure

	/**
	 ********************************** 
	 * Compute distance between instances.
	 ********************************** 
	 */
	public double distance(int paraI, int paraJ) {
		double tempDistance = 0;
		switch (distanceMeasure) {
		case EUCLIDIAN:
			tempDistance = euclidian(paraI, paraJ);
			break;
		case MANHATTAN:
			tempDistance = manhattan(paraI, paraJ);
			break;
		}// Of switch

		return tempDistance;
	}// Of distance

	/**
	 ********************************** 
	 * Compute distance between instances.
	 ********************************** 
	 */
	public double distanceManhattan(int paraI, int paraJ) {
		double tempDistance = 0;
		int tempNumAttributes = numAttributes();
		if (dataType == DECISION_SYSTEM) {
			tempNumAttributes--;
		}// Of if
		for (int i = 0; i < tempNumAttributes; i++) {
			tempDistance += Math.abs(instance(paraI).value(i)
					- instance(paraJ).value(i));
		}// Of for i

		return tempDistance;
	}// Of distance

	/**
	 ********************************** 
	 * Euclidian distance.
	 ********************************** 
	 */
	public double euclidian(int paraI, int paraJ) {
		double tempDistance = 0;
		double tempValue;
		int tempNumAttributes = numAttributes();
		if (dataType == DECISION_SYSTEM) {
			tempNumAttributes--;
		}// Of if
		for (int i = 0; i < tempNumAttributes; i++) {
			tempValue = (instance(paraI).value(i) - instance(paraJ).value(i));
			tempDistance += tempValue * tempValue;
		}// Of for i

		tempDistance = Math.sqrt(tempDistance);

		return tempDistance;
	}// Of euclidian

	/**
	 ********************************** 
	 * Manhattan distance.
	 ********************************** 
	 */
	public double manhattan(int paraI, int paraJ) {
		double tempDistance = 0;
		int tempNumAttributes = numAttributes();
		if (dataType == DECISION_SYSTEM) {
			tempNumAttributes--;
		}// Of if
		for (int i = 0; i < tempNumAttributes; i++) {
			tempDistance += Math.abs(instance(paraI).value(i)
					- instance(paraJ).value(i));
		}// Of for i

		return tempDistance;
	}// Of manhattan

	/**
	 ********************************** 
	 * Manhattan distance of the data set.
	 ********************************** 
	 */
	public double[][] manhattanData() {
		double[][] tempDistance = new double[numInstances()][numInstances()];

		for (int i = 0; i < numInstances(); i++) {

			for (int j = 0; j < numInstances(); j++) {

				tempDistance[i][j] = distanceManhattan(i, j);

			} // of for j
		}// of for i

		return tempDistance;

	}// Of manhattan

	/**
	 ********************************** 
	 * Compute rho
	 ********************************** 
	 */
	public void computeRho() {
		rho = new double[numInstances()];

		for (int i = 0; i < numInstances() - 1; i++) {
			for (int j = i + 1; j < numInstances(); j++) {
				if (distance(i, j) < dc) {
					rho[i] = rho[i] + 1;
					rho[j] = rho[j] + 1;
				}// Of if
			}// Of for j
		}// Of for i

	}// Of computeRho

	public void computeDelta() {
		delta = new double[numInstances()];
		master = new int[numInstances()];
		Arrays.fill(master, -1);
		ordrho = new int[numInstances()];

		ordrho = mergeSortToIndices(rho);
		System.out.println("ordrho:" +Arrays.toString(ordrho));
		delta[ordrho[0]] = maximalDistance;

		for (int i = 1; i < numInstances(); i++) {
			delta[ordrho[i]] = maximalDistance;
			for (int j = 0; j <= i - 1; j++) {
				if (manhattan(ordrho[i], ordrho[j]) < delta[ordrho[i]]) {
					delta[ordrho[i]] = manhattan(ordrho[i], ordrho[j]);
					master[ordrho[i]] = ordrho[j];
				}// of if
			}// of for j
		}// of for i

	}// of computeDelta

	/**
	 ********************************** 
	 * Compute priority. Element with higher priority is more likely to be
	 * selected as a cluster center. Now it is rho * delta. It can also be
	 * rho^alpha * delta.
	 ********************************** 
	 */
	public void computePriority() {
		priority = new double[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			priority[i] = rho[i] * delta[i];
		}// Of for i
	}// Of computePriority

	/**
	 ********************************** 
	 * Compute centers
	 ********************************** 
	 */
	public void computeCenters(int paraNumCenters) {
		centers = new int[paraNumCenters + 1];
		double[] tempWeightedArray = new double[paraNumCenters + 1];
		double tempValue;

		for (int i = 0; i < numInstances(); i++) {
			tempValue = rho[i] * delta[i];
			for (int j = 0; j < paraNumCenters; j++) {
				if (tempValue > tempWeightedArray[j]) {
					// Move the others
					for (int k = paraNumCenters; k > j; k--) {
						centers[k] = centers[k - 1];
						tempWeightedArray[k] = tempWeightedArray[k - 1];
					}// Of for k

					// Insert here
					// //System.out.print("Insert at " + j + " ");
					centers[j] = i;
					tempWeightedArray[j] = tempValue;

					// Already inserted
					break;
				}// Of if
			}// Of for j
		}// Of for i
	}// Of computeCenters

	/**
	 ********************************** 
	 * Compute centers
	 ********************************** 
	 */
	public void computeCentersWM(int paraNumCenters) {
		centers = new int[paraNumCenters];
		int[] priorityIndex = new int[numInstances()];

		priorityIndex = mergeSortToIndices(priority);
		// System.out.println("this is the test");
		// System.out.println("priorityIndex" + Arrays.toString(priorityIndex));
		for (int i = 0; i < paraNumCenters; i++) {
			centers[i] = priorityIndex[i];
		}// of for i
			// System.out.println("centers" + Arrays.toString(centers));
	}// Of computeCenters

	/**
	 ********************************** 
	 * Compute block information
	 ********************************** 
	 */
	public int[][] computeBlockInformation() {
		int tempBlocks = centers.length;
		blockInformation = new int[tempBlocks][];

		for (int i = 0; i < tempBlocks; i++) {
			// Scan to see how many elements
			int tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (clusterIndices[j] == centers[i]) {
					tempElements++;
				}// Of if
			}// Of for k

			// Copy to the list
			blockInformation[i] = new int[tempElements];
			tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (clusterIndices[j] == centers[i]) {
					blockInformation[i][tempElements] = j;
					tempElements++;
				}// Of if
			}// Of for k
		}// Of for i

		return blockInformation;
	}// Of computeBlockInformation

	/**
	 ********************************** 
	 * Compute classification accuracy.
	 ********************************** 
	 */
	/**
	 * public double computeClassificationAccuracy() throws Exception { if
	 * (dataType != DECISION_SYSTEM) { throw new Exception(
	 * "Classification is inapplicable for data type: " + dataType); }// Of if
	 * 
	 * int tempCenters = centers.length - 1; int tempClassLabels =
	 * attribute(numAttributes() - 1).numValues(); int[][]
	 * tempClusterClassMatrix = new int[tempCenters][tempClassLabels];
	 * 
	 * int tempClusterCenter, tempClassLabel; for (int i = 0; i <
	 * numInstances(); i++) { tempClusterCenter = clusterIndices[i];
	 * tempClassLabel = (int) instance(i).value(numAttributes() - 1); for (int j
	 * = 0; j < tempCenters; j++) { if (tempClusterCenter == centers[j]) {
	 * tempClusterClassMatrix[j][tempClassLabel]++; }// Of if }// Of for j }//
	 * Of for i
	 * 
	 * // System.out.println(Arrays.deepToString(tempClusterClassMatrix));
	 * 
	 * double tempCorrect = 0; double tempCurrentClusterMaximal; for (int i = 0;
	 * i < tempCenters; i++) { tempCurrentClusterMaximal = 0; for (int j = 0; j
	 * < tempClassLabels; j++) { if (tempCurrentClusterMaximal <
	 * tempClusterClassMatrix[i][j]) { tempCurrentClusterMaximal =
	 * tempClusterClassMatrix[i][j]; }// Of if }// Of for j
	 * 
	 * tempCorrect += tempCurrentClusterMaximal; }// Of for i
	 * 
	 * return tempCorrect / numInstances(); }// Of computeClassificationAccuracy
	 * 
	 * /**
	 ********************************** 
	 * Cluster according to the centers
	 ********************************** 
	 */
	public void clusterWithCenters() {
		clusterIndices = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			clusterIndices[i] = -1;
		}// Of for i

		for (int i = 0; i < centers.length - 1; i++) {
			clusterIndices[centers[i]] = centers[i];
		}// Of for i

		// //System.out.println("clusterWithCenters 1");
		int[] tempPathIndices = new int[numInstances()];
		int tempPathLength = 0;
		int tempCurrentIndex;

		for (int i = 0; i < numInstances(); i++) {
			// //System.out.println("clusterWithCenters 2.1");
			// Already processed.
			if (clusterIndices[i] != -1) {
				continue;
			}// Of if

			tempCurrentIndex = i;
			tempPathLength = 0;

			while (clusterIndices[tempCurrentIndex] == -1) {
				tempPathIndices[tempPathLength] = tempCurrentIndex;
				tempCurrentIndex = master[tempCurrentIndex];
				tempPathLength++;
			}// Of while

			// Set master for the path
			for (int j = 0; j < tempPathLength; j++) {
				clusterIndices[tempPathIndices[j]] = clusterIndices[tempCurrentIndex];
			}// Of for j
		}// Of for i
	}// Of clusterWithCenters

	/**
	 ********************************** 
	 * Cluster according to the centers
	 ********************************** 
	 */
	public void clusterWithCentersWM(int paraLength) {

		int tempBlocks = paraLength;
		int[] cl = new int[numInstances()];
		int[] ordrho = new int[numInstances()];
		ordrho = mergeSortToIndices(rho);
		
		clusterIndices = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			cl[i] = -1;
		}// of for i

		int tempNumber = 0;
		int Ncluster = 0;
		for (int i = 0; i < numInstances(); i++) {
			if (tempNumber < tempBlocks) {
				cl[centers[i]] = Ncluster;
				Ncluster++;
				tempNumber++;
			}// of if
		}// of for i
			// System.out.println("master" + Arrays.toString(master));
		
		// System.out.println("centers" + Arrays.toString(centers));
		// System.out.println("This is the test 1");
		// System.out.println("cl" + Arrays.toString(cl));
		for (int i = 0; i < numInstances(); i++) {
			// System.out.println(i);
			if (cl[ordrho[i]] == -1) {
				// System.out.println("This is the test 1.1");
				cl[ordrho[i]] = cl[master[ordrho[i]]];

				// System.out.println("This is the test 1.2");
				// System.out.println("rhoIndexArray"+Arrays.toString(ordrho)+"Now we will label"+i+"labelArray"+Arrays.toString(cl));
				// System.out.println("This is the test 1.2");
			}// of if
		}// of for i
			// System.out.println("This is the test 2");
		for (int i = 0; i < numInstances(); i++) {
			clusterIndices[i] = centers[cl[i]];
		}

		// System.out.println("clusterIndices" +
		// Arrays.toString(clusterIndices));
	}// Of clusterWithCenters

	/**
	 ********************************** 
	 * Compute the maximal of a array.
	 ********************************** 
	 */
	public static int getMax(int[] arr) {

		int max = arr[0];

		for (int x = 1; x < arr.length; x++) {
			if (arr[x] > max)
				max = arr[x];

		}
		return max;

	}

	/**
	 ********************************** 
	 * Compute the maximal index of a array.
	 ********************************** 
	 */

	public int getMaxIndex(int[] paraArray) {
		int maxIndex = 0;
		int tempIndex = 0;
		int max = paraArray[0];

		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] > max) {
				max = paraArray[i];
				tempIndex = i;
			}// of if
		}// of for i
		maxIndex = tempIndex;
		return maxIndex;
	}// of getMaxIndex

	/**
	 ********************************** 
	 * Cluster based active learning.
	 * 
	 * @param paraBlockStepLength
	 *            The number of blocks added for each round.
	 * @param paraTeachInstances
	 *            How many instances should be taught for each block.
	 ********************************** 
	 */
	public void clusterBasedActiveLearningWithSorting(int paraBlock,
			int paraTeachEachBlock, int paraTeach) {

		
		predictedLabels = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			predictedLabels[i] = -1;
		}// of for i

		numTeach = 0;
		numPredict = 0;
		numVote = 0;

		int tempBlocks = paraBlock;

		computeRho();
		computeDelta();
		computePriority();
		descendantIndices = mergeSortToIndices(priority);
		alreadyClassified = new boolean[numInstances()];

		while (true) {
			// {
			System.out.println(tempBlocks);
			computeCentersWM(tempBlocks);
			System.out.println("centers:" + Arrays.toString(centers));
			clusterWithCentersWM(tempBlocks);
			System.out.println("clusterIndices:" +Arrays.toString(clusterIndices));
			computeBlockInformation();
			
			System.out.println("blockInfo");
			for (int i = 0; i < blockInformation.length; i++) {
				System.out.println(Arrays.toString(blockInformation[i]));
			}
			
			boolean[] tempBlockProcessed = new boolean[tempBlocks];
			int tempUnProcessedBlocks = 0;
			for (int i = 0; i < blockInformation.length; i++) {
				tempBlockProcessed[i] = true;
				for (int j = 0; j < blockInformation[i].length; j++) {

					if (!alreadyClassified[blockInformation[i][j]]) {
						tempBlockProcessed[i] = false;
						tempUnProcessedBlocks++;
						break;
					}// of if
				}// of for j
			}// of for i

			System.out.println("tempBlockProcessed " + Arrays.toString(tempBlockProcessed));			
			System.out.println("tempUnprocessedBlocks " + tempUnProcessedBlocks);
			
			
			for (int i = 0; i < blockInformation.length; i++) {
				// Step 2.3.1 如果该块已经被处理完，则直接退出，不需要购买

				if (tempBlockProcessed[i]) {
					continue;
				}// of if


				if (blockInformation[i].length < paraTeachEachBlock) {

					for (int j = 0; j < blockInformation[i].length; j++) {
						if (!alreadyClassified[blockInformation[i][j]]) {
							if (numTeach >= paraTeach) {
								break;
							}// of if
							predictedLabels[blockInformation[i][j]] = (int) instance(
									blockInformation[i][j]).classValue();
							alreadyClassified[blockInformation[i][j]] = true;
							numTeach++;
							System.out.println("numTeach first = " + numTeach);
						}// of if
					}// of for j
				}// of if

				
				int[] ordPriority = new int[blockInformation[i].length];

				int tempIndex = 0;
				for (int j = 0; j < numInstances(); j++) {
					if (clusterIndices[descendantIndices[j]] == centers[i]) {
						ordPriority[tempIndex] = descendantIndices[j];
						// tempPriority[tempIndex] = priority[j];
						tempIndex++;
					}// of if
				}// of for j

				int tempNumTeach = 0;
				for (int j = 0; j < blockInformation[i].length; j++) {
					if (alreadyClassified[ordPriority[j]]) {
						continue;
					}// of if
					if (numTeach >= paraTeach) {
						break;
					}// of if
					predictedLabels[ordPriority[j]] = (int) instance(
							ordPriority[j]).classValue();
					alreadyClassified[ordPriority[j]] = true;
					numTeach++;

					tempNumTeach++;

					if (tempNumTeach >= paraTeachEachBlock) {
						break;
					}// of if

				}// of for j

			} // of for i

			
			boolean tempPure = true;

			for (int i = 0; i < blockInformation.length; i++) {

				
				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				boolean tempFirstLable = true;
				
				int tempCurrentInstance;
				int tempLable = 0;

				for (int j = 0; j < blockInformation[i].length; j++) {
					tempCurrentInstance = blockInformation[i][j];
					if (alreadyClassified[tempCurrentInstance]) {

						if (tempFirstLable) {
							tempLable = predictedLabels[tempCurrentInstance];
							tempFirstLable = false;
						} else {
							if (tempLable != predictedLabels[tempCurrentInstance]) {
								tempPure = false;
								break;
							}// of if
						} // of if
					}// of if
				}// of for j

				System.out.println("already classified: " + Arrays.toString(alreadyClassified));
				if (tempPure) {
					for (int j = 0; j < blockInformation[i].length; j++) {
						if (!alreadyClassified[blockInformation[i][j]]) {
							predictedLabels[blockInformation[i][j]] = tempLable;
							alreadyClassified[blockInformation[i][j]] = true;
							numPredict++;
						}// of if
					}// of for j
				}// of if
			}// of for i

			tempBlocks++;

			
			if (tempUnProcessedBlocks == 0) {
				break;
			}// of if

			
			if (numTeach >= paraTeach) {
				break;
			}// of if

		} // of while

		int max = getMax(predictedLabels);
		computeCentersWM(max + 1);
		clusterWithCentersWM(max + 1);
		computeBlockInformation();
		
		int[][] vote = new int[max + 1][max + 1];
		int voteIndex = -1;

		for (int i = 0; i < blockInformation.length; i++) {

			
			for (int j = 0; j < blockInformation[i].length; j++) {

				for (int k = 0; k <= max; k++) {

					if (predictedLabels[blockInformation[i][j]] == k) {
						vote[i][k]++;
					}// of if
				}// of for k
			}// of for j

			
			voteIndex = getMaxIndex(vote[i]);

			

			for (int j = 0; j < blockInformation[i].length; j++) {
				if (predictedLabels[blockInformation[i][j]] == -1) {
					predictedLabels[blockInformation[i][j]] = voteIndex;
					numVote++;
				}// of if
			}// of for j
		}// of for i

		System.out.println("clusterBasedActiveLearning finish!");
		System.out.println("numTeach = " + numTeach + "; predicted = "
				+ numPredict + "; numVote = " + numVote);
		System.out.println("PredictedLabels : " + Arrays.toString(predictedLabels));
		System.out.println("Accuracy = " + getPredictionAccuracy());
		

	}// of activeLearing

	/**
	 ******************* 
	 * Get prediction accuracy.
	 ******************* 
	 */
	public double getPredictionAccuracy() {
		double tempInCorrect = 0;
		// System.out.println("Incorrectly classified instances:");
		for (int i = 0; i < numInstances(); i++) {
			if (predictedLabels[i] != (int) instance(i).classValue()) {
				tempInCorrect++;
				System.out.print("" + i + ", ");
			}// Of if
		}// Of for i
		System.out.println();
		System.out.println("This is the incorrect:\r\n" + tempInCorrect);

		return (numInstances() - numTeach - tempInCorrect)
				/ (numInstances() - numTeach);

	}// Of getPredictionAccuracy

	@Override
	public String toString() {
		return "Density [distanceMeasure=" + distanceMeasure + ",\r\n dataType=" + dataType + ",\r\n numTeach=" + numTeach
				+ ",\r\n numPredict=" + numPredict + ",\r\n numVote=" + numVote + ",\r\n dc=" + dc + ",\r\n ordrho="
				+ Arrays.toString(ordrho) + ",\r\n rho=" + Arrays.toString(rho) + ",\r\n delta=" + Arrays.toString(delta)
				+ ",\r\n priority=" + Arrays.toString(priority) + ",\r\n centers=" + Arrays.toString(centers)
				+ ",\r\n maximalDistance=" + maximalDistance + ",\r\nmaximalDelta=" + maximalDelta + ",\r\nclusterIndices="
				+ Arrays.toString(clusterIndices) + ",\r\nblockInformation=" + Arrays.toString(blockInformation)
				+ ",\r\nmaster=" + Arrays.toString(master) + ",\r\npredictedLabels=" + Arrays.toString(predictedLabels)
				+ ",\r\nalreadyClassified=" + Arrays.toString(alreadyClassified) + ",\r\ndescendantIndices="
				+ Arrays.toString(descendantIndices) + ",\r\nteachInstancesForEachBlock=" + teachInstancesForEachBlock
				+ "]";
	}

	/**
	 ******************* 
	 * Density test.
	 ******************* 
	 */
	public static void densityTest() {
		totalRuntime = 0;
		String arffFilename = "src/data/arff/iris2.arff";

		try {
			FileReader fileReader = new FileReader(arffFilename);
			Density tempData = new Density(fileReader);
			fileReader.close();
			// tempData.setClassIndex(tempData.numAttributes() - 1);
			tempData.setDistanceMeasure(MANHATTAN);// EUCLIDIAN MANHATTAN
			// System.out.println("This is the data:\r\n" + tempData);
			
			tempData.setDcFromMaximalDistance(0.5);
			tempData.clusterBasedActiveLearningWithSorting(2, 2, 20);
			System.out.println("rho is " + Arrays.toString(tempData.rho));
			System.out
					.println("master are " + Arrays.toString(tempData.master));
			 System.out.println("delta are " +
			 Arrays.toString(tempData.delta));
			 System.out.println("priority are "
			 + Arrays.toString(tempData.priority));
			 
			 System.out.println(tempData);
		} catch (Exception ee) {
			// System.out.println("Error occurred while trying to read \'"
			// + arffFilename + "\' in densityTest().\r\n" + ee);
		}// Of try
	}// Of densityTest

	/**
	 ********************************** 
	 * Generate a random sequence of [0, n - 1].
	 * 
	 * @author Hengru Zhang, Revised by Fan Min 2013/12/24
	 * 
	 * @param paraLength
	 *            the length of the sequence
	 * @return an array of non-repeat random numbers in [0, paraLength - 1].
	 ********************************** 
	 */
	public static int[] generateRandomSequence(int paraLength) {
		Random random = new Random();
		// Initialize
		int[] tempResultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			tempResultArray[i] = i;
		}// Of for i

		// Swap some elements
		int tempFirstIndex, tempSecondIndex, tempValue;
		for (int i = 0; i < paraLength / 2; i++) {
			tempFirstIndex = random.nextInt(paraLength);
			tempSecondIndex = random.nextInt(paraLength);
			
			// Really swap elements in these two indices
			tempValue = tempResultArray[tempFirstIndex];
			tempResultArray[tempFirstIndex] = tempResultArray[tempSecondIndex];
			tempResultArray[tempSecondIndex] = tempValue;
		}// Of for i

		return tempResultArray;
	}// Of generateRandomSequence

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged.<br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2].<br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @author Fan Min 2016/09/09
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */

	public static int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];// 两个维度交换存储排序tempIndex控制

		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i
			// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;
		while (tempCurrentLength < tempLength) {

			// Divide into a number of groups
			// Here the boundary is adaptive to array length not equal to 2^k.

			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {
				// Boundaries of the group

				tempFirstStart = i * tempCurrentLength * 2;

				tempSecondStart = tempFirstStart + tempCurrentLength;

				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {
					tempSecondEnd = tempLength - 1;
				} // Of if

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;

				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;
					} // Of for j
					break;
				} // Of if

				while ((tempFirstIndex <= tempSecondStart - 1)
						&& (tempSecondIndex <= tempSecondEnd)) {

					if (paraArray[resultMatrix[tempIndex % 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex % 2][tempSecondIndex]]) {

						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][tempFirstIndex];
						int a = (tempIndex + 1) % 2;

						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][tempSecondIndex];
						int b = (tempIndex + 1) % 2;

						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;

				} // Of while

				// Remaining part

				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];

					tempCurrentIndex++;

				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];

					tempCurrentIndex++;
				} // Of for j

			} // Of for i

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices

	public static void main(String[] args) {
		long start = System.currentTimeMillis();

//		densityTest();

		long end = System.currentTimeMillis();
		System.out.println("计算花费时间" + (end - start) + "毫秒!");

		// double[] tempArray = {1.2, 2.3, 0.4, 0.5};
		// double[] tempArray = {3.1, 5.2, 6.3, 2.1, 4.4};
		// int[] tempSortedIndices = mergeSortToIndices(tempArray);
		// System.out.println("The indices are: " +
		// Arrays.toString(tempSortedIndices));
		
		do {
			int i = 0;
		} while (0 < 1);
		
		
	}// Of main
}// Of Density

