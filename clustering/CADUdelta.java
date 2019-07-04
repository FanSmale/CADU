package fansactive.clustering;

import java.io.FileReader;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import weka.core.DistanceFunction;
import weka.core.Instances;

/**
 * CADU algorithm implementation. 
 */
public class CADUdelta {

	double[][] data;

	// double[][] distance;

	int[] rLabel;

	int[] master;

	double[] delta;

	double[] rho;

	double dc;

	// mCost[0] = m(-, +) mCost[1] = m(+, -)
	double[] mCost;

	double tCost;

	double loose;

	/**
	 * The minimum threshold to queried all instances.
	 */
	double T;

	double maxDistance;

	boolean[] leaf;
	
	int[] prioIndices;
	
	int[] rhoIndices;
	
	int[] predictedLabels;
	
	boolean[] queried;
	
	int blockCount = 0;

	int blockNum = 0;
	
	boolean[] processed;
	
	int[] blockInfo;
	
	// Two temp variables stored optimal R and optimal B. When a leaf node is split, tR and tB are not change.  
	int tR = -1;
	
	int tB = -1;
	
	
	public CADUdelta(double[][] pData, int[] pRLabels, double pDc, double[] pMCost, double pTCost, double pLoose) {
		rLabel = pRLabels;
		data = pData;
		dc = pDc;
		mCost = pMCost;
		loose = pLoose;
		//T = pT;
		tCost = pTCost;
	}// Of CADU3

	public void costSensitiveACLearning() {

		// computeDistance();
		queried = new boolean[data.length];
		predictedLabels = new int[data.length];
		Arrays.fill(predictedLabels, -1);

		int[] pPrioInd = new int[data.length];
		for (int i = 0; i < pPrioInd.length; i++) {
			pPrioInd[i] = i;
		}// Of for i
		
		processed = new boolean[data.length];
		blockInfo = new int[data.length];
		Arrays.fill(blockInfo, -1);
		
		int[] deltaIndices = argSort(delta, Order.DESC);
		
		if(master[deltaIndices[0]] != -1) {
			for (int i = 0; i < deltaIndices.length; i++) {
				if (master[deltaIndices[i]] == -1) {
					int tmpIndex = deltaIndices[0];
					deltaIndices[0] = deltaIndices[i];
					deltaIndices[i] = tmpIndex;
					break;
				}
			}
		}
		
		// blockLearning(pPrioInd);
		blockLearning(deltaIndices);
	}// Of costSensitiveACLearning

	/**
	 ********************* 
	 * Block learning
	 ********************* 
	 */
	private void blockLearning(int[] pPrioInd) {
		blockCount++;
		//System.out.print("Learning with " + pPrioInd.length + " instances,");
		// 1. Stop criterion
		// 1.1 All are labeled
		boolean allLabeled = true;
		for (int i = 0; i < pPrioInd.length; i++) {
			if (predictedLabels[pPrioInd[i]] == -1) {
				allLabeled = false;
				break;
			}// Of if
		}// Of for i
		if (allLabeled) {
//			System.out.println("This block is all labeled.");
			return;
		}// Of if

		// 1.2 This block is too small and queried all
//		if (pPrioInd.length <= T) {
////			System.out.println("This block is too small, label all.");
//			for (int i = 0; i < pPrioInd.length; i++) {
//				if (predictedLabels[pPrioInd[i]] == -1) {
//					queried[pPrioInd[i]] = true;
//					predictedLabels[pPrioInd[i]] = rLabel[pPrioInd[i]];
//				}// Of if
//			}// Of for i
//			return;
//		}// Of if

		// System.out.println(Arrays.toString(pPrioInd));
		// 1.3 If this block is not pure.
		boolean[] notPure = new boolean[2];
		for (int i = 1; i < pPrioInd.length; i++) {
			if (predictedLabels[pPrioInd[i]] == 0) {
				notPure[0] = true;
			} else if (predictedLabels[pPrioInd[i]] == 1) {
				notPure[1] = true;
			}// Of if
			// Split
			if (notPure[0] && notPure[1]) {
//				System.out.println("This block is already impure, split it now: \r\n" + Arrays.toString(pPrioInd));
				splitLearning(pPrioInd, i);
				return;
			}// Of if
		}// Of for i

		// 2. Query a part of this block
		// 2.0 Query the first instance.
		if (!queried[pPrioInd[0]]) {
			predictedLabels[pPrioInd[0]] = rLabel[pPrioInd[0]];
			queried[pPrioInd[0]] = true;
		}// Of if

		// 2.1 // Look up
		int[] RandB = null;
		if (tR != -1 && tB != -1) {
			RandB = new int[]{tR, tB};
		} else {
			RandB = lookup(pPrioInd.length);
			tR = RandB[0];
			tB = RandB[1];
		}// Of if
		
//		System.out.print(" Looking up " + pPrioInd.length + " Instances: " + Arrays.toString(RandB) + "  ");
		// How many instances is queried?
		int lenQueried = RandB[predictedLabels[pPrioInd[0]]];
		
		// 2.2. Query instances
		// There are some cases:
		// 2.2.1. Number of already queried is less than min(R, B)
		// 2.2.2. Number of already queried is greater than min(R, B) and less
		// than max(R, B)
		// 2.2.3. Number of already queried is greater than max(R, B)
		// We consider all these cases.

		for (int i = 1; i < lenQueried && i < pPrioInd.length; i++) {
			queried[pPrioInd[i]] = true;
			predictedLabels[pPrioInd[i]] = rLabel[pPrioInd[i]];
			if (predictedLabels[pPrioInd[i]] != predictedLabels[pPrioInd[0]]) {
//				System.out.println("Impure block, split now." + Arrays.toString(pPrioInd));
				splitLearning(pPrioInd, i);
				return;
			}// Of if
		}// Of for i

		// Predict
//		System.out.println("This block is pure, predict others. " + Arrays.toString(pPrioInd));
//		System.out.println("The number of instances is " + pPrioInd.length);
		if (pPrioInd.length > maxsizePureBlock) {
			maxsizePureBlock = pPrioInd.length; 
		}// Of if
		
		for (int i = lenQueried; i < pPrioInd.length; i++) {
			predictedLabels[pPrioInd[i]] = predictedLabels[pPrioInd[0]];
		}// Of for i
	}// Of blockLearning
	
	int maxsizePureBlock = Integer.MIN_VALUE;
	
	/**
	 ********************* 
	 * Look up optimal R and B
	 ********************* 
	 */
	private int[] lookup(int pSize) {
		// Linear Search
		double[] tmpMinCost = new double[] { 0.5 * mCost[0] * pSize, 0.5 * mCost[1] * pSize };
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		ra[0] = 0.5;
		double[] tmpCost = new double[2];
		for (int i = 1; i <= pSize; i++) {
			ra[i] = expectPosNum(i, 0, pSize) / pSize;
			for (int j = 0; j < 2; j++) {
				tmpCost[j] = (1 - ra[i]) * mCost[j] * pSize + tCost * i;
				if (tmpCost[j] < tmpMinCost[j]) {
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;
					if (i == pSize) {
						Arrays.fill(isFind, true);
					}// Of if
				} else {
					isFind[j] = true;
				}// Of if
			}// Of for j
			//System.out.println("QueriedInstance: " + i + ", cost: " + tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {
				Arrays.fill(isFind, false);
				for (int j = 0; j <= i; j++) {
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * mCost[k] * pSize + tCost * j;
						if (tmpCost[k] <= (1.000001 + loose) * tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j;
						}// Of if
						if (isFind[0] && isFind[1]) {
							return starLoose;
						}// Of if
					}// Of for k
				}// Of for j
			}// Of if
		}// Of for i 
		return new int[] { 0, 0 };
		// throw new RuntimeException("Error occured in lookup("+pSize+")");
	}// Of lookup

	/**
	 ********************* 
	 * Split a block and do blockLearning again.
	 ********************* 
	 */
	private void splitLearning(int[] sortedBlock, int paraSeparatedIndex) {
		
		if (leaf[sortedBlock[paraSeparatedIndex]]) {
//			System.out.println("Leaf: " + sortedBlock[paraSeparatedIndex]);
			int[] sortedBlocksub1 = new int[sortedBlock.length-1];
			System.arraycopy(sortedBlock, 0, sortedBlocksub1, 0, paraSeparatedIndex);
			System.arraycopy(sortedBlock, paraSeparatedIndex + 1, sortedBlocksub1, paraSeparatedIndex, 
					sortedBlock.length - paraSeparatedIndex - 1);
			blockLearning(sortedBlocksub1);
			return;
		}// Of if
//		System.out.println(sortedBlock[0] + "\t" + sortedBlock[paraSeparatedIndex]);
		tR = -1;
		tB = -1;
		processed[sortedBlock[0]] = true;
		processed[sortedBlock[paraSeparatedIndex]] = true;
		blockInfo[sortedBlock[0]] = blockNum;
		blockInfo[sortedBlock[paraSeparatedIndex]] = blockNum + 1;
		int[] belongsTo = new int[sortedBlock.length];
		for (int i = 0; i < sortedBlock.length; i++) {
			belongsTo[i] = assignLabel(sortedBlock[i]) - blockNum;
		}// Of for i
		int[] count = new int[2];
		for (int i = 0; i < sortedBlock.length; i++) {
			count[belongsTo[i]]++;
		}// Of for i
		blockNum++;
		int[][] twoBlocks = new int[2][];
		twoBlocks[0] = new int[count[0]];
		twoBlocks[1] = new int[count[1]];
		Arrays.fill(count, 0);
		for (int i = 0; i < sortedBlock.length; i++) {
			twoBlocks[belongsTo[i]][count[belongsTo[i]]++] = sortedBlock[i];
			processed[sortedBlock[i]] = false;
		}// Of for i
		blockLearning(twoBlocks[0]);
		tR = -1;
		tB = -1;
		blockLearning(twoBlocks[1]);
		tR = -1;
		tB = -1;
	}// Of splitLearning

	/**
	 ********************* 
	 * Assign label for some instances.
	 ********************* 
	 */
	private int assignLabel(int pIndex) {
		if (!processed[pIndex]) {
			int tempLabel = assignLabel(master[pIndex]);
			processed[pIndex] = true;
			blockInfo[pIndex] = tempLabel;
			return tempLabel;
		}// Of if
		return blockInfo[pIndex];
	}// Of assignLabel

	/**
	 * Euclidean Distance public void computeDistance() { double[][] tDistance =
	 * new double[data.length][data.length];
	 * 
	 * for (int i = 0; i < data.length; i++) { for (int j = i + 1; j <
	 * data.length; j++) { double t = 0; for (int k = 0; k < data[0].length;
	 * k++) { // t += (data[i][k] - data[j][k]) * (data[i][k] - // data[j][k]);
	 * t += Math.abs(data[i][k] - data[j][k]); } tDistance[i][j] = t;
	 * tDistance[j][i] = t; } } distance = tDistance; }
	 */

	/**
	 ********************* 
	 * Compute Manhattan Distance
	 ********************* 
	 */
	public double manhattan(int paraI, int paraJ) {
		double tDistance = 0;
		for (int i = 0; i < data[0].length; i++) {
			tDistance += Math.abs(data[paraI][i] - data[paraJ][i]);
		}// Of for i
		return tDistance;
	}// of manhattan

	/**
	 ********************* 
	 * Compute Euclidean distance
	 ********************* 
	 */
	public double distance(int paraI, int paraJ) {
		double tDistance = 0;
		for (int i = 0; i < data[0].length; i++) {
			tDistance += (data[paraI][i] - data[paraJ][i]) * (data[paraI][i] - data[paraJ][i]);
		}// Of for i
		return Math.sqrt(tDistance);
	}//Of distance
	
	/**
	 *********************
	 * Compute Maximum distance
	 ********************* 
	 */
	public void computeMaxDistance() {
		double tMaxDistance = Double.MIN_VALUE;
		double tempDistance = 0;
		for (int i = 0; i < data.length; i++) {
			for (int j = i + 1; j < data.length; j++) {
				tempDistance = distance(i, j);
				if (tMaxDistance < tempDistance) {
					tMaxDistance = tempDistance;
				}// Of if
			}// Of for j
		}// Of for i
		maxDistance = tMaxDistance;
	}// Of computeMaxDistance

	/**
	 *********************
	 * Compute Rho
	 *********************
	 */
	public void computeRho() {
		double[] tRho = new double[data.length];
		double dcDist = dc * maxDistance;

		for (int i = 0; i < data.length; i++) {
			for (int j = i + 1; j < data.length; j++) {
				if (distance(i, j) < dcDist) {
					tRho[i]++;
					tRho[j]++;
				}// Of if
			}// Of for j
		}// Of for i
		rho = tRho;
	}// Of computeRho

	/**
	 *********************
	 * Sort rho and obtain the indices in descending order.
	 ********************* 
	 */
	public void computeRhoIndices() {
		rhoIndices = argSort(rho, Order.DESC);
	}// Of computeRhoIndices

	/**
	 *********************
	 * Compute the total cost.
	 *********************
	 */
	public double totalCost() {
		double sum = 0;
		for (int i = 0; i < data.length; i++) {
			if (queried[i]) {
				sum += tCost;
			} else {
				// Actual positive; Predict negative
				if (predictedLabels[i] == 0 && rLabel[i] == 1) {
					sum += mCost[0];
				} else if (predictedLabels[i] == 1 && rLabel[i] == 0) {
					sum += mCost[1];
				}// Of if
			}// Of if
		}// Of for i
//		System.out.println("tempQuried = " + tempQuried);
		return sum;
	}// Of totalCost

	/**
	 *********************
	 * Compute delta.
	 *********************
	 */
	public void computeDelta() {
		double[] tDelta = new double[data.length];
		int[] tMaster = new int[data.length];
		Arrays.fill(tMaster, -1);
		Arrays.fill(tDelta, maxDistance);

		for (int i = 1; i < data.length; i++) {
			double tMinDist = maxDistance;
			int tMinIndex = 0;
			double tempDistance = 10000;
			for (int j = 0; j < i; j++) {
				tempDistance = distance(rhoIndices[i], rhoIndices[j]);
				if (tempDistance < tMinDist) {
					tMinDist = tempDistance;
					tMinIndex = j;
				}// Of if
			}// Of for j
			tDelta[rhoIndices[i]] = tMinDist;
			tMaster[rhoIndices[i]] = rhoIndices[tMinIndex];
		}// Of for i
		delta = tDelta;
		master = tMaster;
	}// Of computeDelta

	/**
	 *********************
	 * Compute priority.
	 ********************* 
	 */
	public void computePrio() {
		double[] tmpProduct = new double[data.length];
		for (int i = 0; i < tmpProduct.length; i++) {
			tmpProduct[i] = rho[i] * delta[i];
		}// Of for i
		prioIndices = argSort(tmpProduct, Order.DESC);
	}// Of computePrio

	/**
	 *********************
	 * Save the leaf node
	 ********************* 
	 */
	public void computeLeaf() {
		Set<Integer> set = new HashSet<>();
		for (int i = 0; i < master.length; i++) {
			if (master[i] == -1) {
				set.add(i);
			} else {
				set.add(master[i]);
			}// Of if
		}// Of for i
		leaf = new boolean[data.length];
		int count = 0;
		for (int i = 0; i < master.length; i++) {
			if (!set.contains(i)) {
				leaf[i] = true;
			}// Of if
		}// Of for i
	}// Of computeLeaf
	
	// Sorting order
	public enum Order {
		ASC, DESC
	}// Of Order

	/**
	 * sort an array but return the indices. E.g. by given array [3, 4, 2, 1],
	 * this method will return [3, 2, 0, 1]. If it is descending order, it will
	 * return [1, 0, 2, 3]
	 * 
	 * @param valueArray
	 *            The given value array.
	 * @param o
	 *            If the sort is ascending or descending.
	 * @return the indices of sorted array.
	 */
	public static int[] argSort(double[] valueArray, Order o) {
		Map.Entry<Integer, Double>[] entries = new MyEntry[valueArray.length];
		for (int i = 0; i < entries.length; i++) {
			entries[i] = new MyEntry(i, o == Order.DESC ? -valueArray[i] : valueArray[i]);
		}// Of for i
		// lambda function
		Arrays.sort(entries, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
		int[] indices = new int[valueArray.length];
		for (int i = 0; i < valueArray.length; i++) {
			indices[i] = entries[i].getKey();
		}// Of for i
		return indices;
	}// Of argSort

	public static class MyEntry<K, V> implements Map.Entry<K, V> {
		K key;
		V value;

		public MyEntry(K key, V value) {
			super();
			this.key = key;
			this.value = value;
		}// Of MyEntry

		@Override
		public V getValue() {
			return value;
		}// Of getValue

		@Override
		public V setValue(V value) {
			V v = this.value;
			this.value = value;
			return v;
		}// Of setValue

		@Override
		public K getKey() {
			return key;
		}// Of getKey

		@Override
		public String toString() {
			return "(" + key + ", " + value + ")";
		}// Of toString
	}// Of MyEntry

	public static void test() throws Exception {
		// Instances instances = new Instances(new
		// FileReader("src/data/iris.arff"));
		// Instances instances = new Instances(new FileReader(
		// "src/data/breast-cancer_test.arff"));
		String[] strings = new String[]{"breast-cancer_test", "flare-solar_test_5000", "german_test_5000", "heart_test_5000", "image_test_5000", "splice_test_5000", "thyroid_test", "titanic_test_5000"};
		for (String ds:strings) {
			double dc = 0.05;
			for (; dc < 0.601; dc += 0.05) {
				Instances instances = new Instances(new FileReader("src/data/arff/"+ds+".arff"));
				// System.out.print("[");
				// for (int i = 0; i < instances.numInstances(); i++) {
				// System.out.print("[");
				// for (int j = 0; j < instances.instance(i).numAttributes()-1; j++) {
				// System.out.print(instances.instance(i).value(j)+"\t");
				// }
				// System.out.print("];");
				// }
				// System.out.println("]");
				// System.out.print("[");
				// for (int i = 0; i < instances.numInstances(); i++) {
				// System.out.print((int)instances.instance(i).value(instances.numAttributes()-1)
				// + "\t");
				// }
				// System.out.println("]");

				double[][] data = new double[instances.numInstances()][instances.numAttributes() - 1];
				int[] rLabels = new int[data.length];

				for (int i = 0; i < instances.numInstances(); i++) {
					for (int j = 0; j < instances.numAttributes() - 1; j++) {
						data[i][j] = instances.instance(i).value(j);
					}
					rLabels[i] = (int) instances.instance(i).value(instances.numAttributes() - 1);
				}// Of for i

				double tCost = 1;

				double loose = 0;

				CADUdelta cadu = new CADUdelta(data, rLabels, dc, new double[] { 4, 9 }, tCost, loose);
				System.out.println(Arrays.toString(cadu.lookup(15)));
				cadu.computeMaxDistance();
				cadu.computeRho();
				cadu.computeRhoIndices();
				cadu.computeDelta();
				cadu.computePrio();
				cadu.computeLeaf();
				
				// CADU cadu2 = new CADU(data, rLabels, dc, 4, 9, tCost, T, loose);
				cadu.costSensitiveACLearning();

				// cadu.computeMaxDistance();
				// cadu.computeRho();

				// System.out.println(Arrays.toString(cadu.rho));

				// cadu2.costSensitiveACLearning();

				// System.out.println(Arrays.equals(cadu.predictedLabels,
				// cadu2.predictedLabels));

//				System.out.println(Arrays.toString(cadu.blockInfo));
//				System.out.println("Total cost:");
				System.out.print(cadu.totalCost() + "\t");
		//
//				System.out.println(Arrays.toString(cadu.master));
				
//				System.out.print(cadu.maxsizePureBlock + "\t");
			}// Of for dc
			System.out.println();
		}
				
	}// Of test

	public static void testSmallData() throws Exception{
		Instances instances = new Instances(new FileReader("src/data/arff/test.arff"));
		// Step 1. Input data
		double[][] data = new double[instances.numInstances()][instances.numAttributes() - 1];
		int[] rLabels = new int[data.length];

		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes() - 1; j++) {
				data[i][j] = instances.instance(i).value(j);
			}
			rLabels[i] = (int) instances.instance(i).value(instances.numAttributes() - 1);
		}// Of for i
//		double[][] data = {
//				{2, 3, 0},
//				{2, -3, 0},
//				{0, 0, 1},
//				{0, 0, -1},
//				
//		};
//		int[] rLabels = {0, 1, 0, 0};
		///////////// Testing pass
		double tCost = 1;
		double loose = 0;
		double dc = 0.4;
		CADUdelta cadu = new CADUdelta(data, rLabels, dc, new double[] { 4, 9 }, tCost, loose);
		cadu.computeMaxDistance();
		System.out.println("MaxDistance is:\t" + cadu.maxDistance);
		cadu.computeRho();
		for (int i = 0; i < cadu.data.length; i++) {
			for (int j = 0; j < cadu.data.length; j++) {
				System.out.printf("%.2f\t", cadu.manhattan(i, j));
			}
			System.out.println();
		}
		System.out.println("DC = " + dc + ", \r\nRho = " + Arrays.toString(cadu.rho));
		
		cadu.computeRhoIndices();
		System.out.println("Rho descend indices: " + Arrays.toString(cadu.rhoIndices));
		cadu.computeDelta();
		System.out.println("Delta: " + Arrays.toString(cadu.delta));
		System.out.println("Master: " + Arrays.toString(cadu.master));
		cadu.computePrio();
		System.out.println("Priority: " + Arrays.toString(cadu.prioIndices));
		cadu.computeLeaf();
		System.out.println("Leaf: " + Arrays.toString(cadu.leaf));
		
		cadu.costSensitiveACLearning();
		System.out.println("Total Cost: " + cadu.totalCost());
		System.out.println("Queried: " + Arrays.toString(cadu.queried));
		
	}
	// @Override
	// public String toString() {
	// StringBuilder sBuilder= new StringBuilder();
	// sBuilder.append("data: \r\n" + Arrays.deepToString(data));
	// sBuilder.append("\r\nrealLabels: \r\n" + Arrays.toString(rLabel));
	// sBuilder.append("\r\nmaster: \r\n" + Arrays.toString(master));
	// sBuilder.append("\r\nrho: \r\n" + Arrays.toString(rho));
	// sBuilder.append("\r\ndelta: \r\n" + Arrays.toString(delta));
	// sBuilder.append("\r\npredictedLabels: \r\n" +
	// Arrays.toString(predictedLabels));
	//
	// return sBuilder.toString();
	// }
	
	public static void test1() throws Exception {
		Instances instances = new Instances(new FileReader("src/data/arff/heart_norepeat.arff"));
		// Step 1. Input data
		double[][] data = new double[instances.numInstances()][instances.numAttributes() - 1];
		int[] rLabels = new int[data.length];
		
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes() - 1; j++) {
				data[i][j] = instances.instance(i).value(j);
			}
			rLabels[i] = (int) instances.instance(i).value(instances.numAttributes() - 1);
		}// Of for i

		for (double i = 0.01; i <= 0.2001; i += 0.01) {
			double tCost = 1;
			double loose = 0;
			double dc = 0.05;
			CADUdelta cadu = new CADUdelta(data, rLabels, dc, new double[] { 4, 9 }, tCost, loose);
			cadu.computeMaxDistance();
			cadu.computeRho();
			cadu.computeRhoIndices();
			cadu.computeDelta();
			System.out.println(Arrays.toString(cadu.master));
			cadu.computePrio();
			cadu.computeLeaf();
			cadu.costSensitiveACLearning();
			System.out.print(cadu.totalCost() + "\t");
//			System.out.println(cadu.blockCount);	
		}// Of for i
	}//Of test1
	
	public static void testFs() throws Exception {
		Instances instances = new Instances(new FileReader("src/data/arff/image_norepeat.arff"));
		// Step 1. Input data
		double[][] data = new double[instances.numInstances()][instances.numAttributes() - 1];
		int[] rLabels = new int[data.length];
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes() - 1; j++) {
				data[i][j] = instances.instance(i).value(j);
			}
			rLabels[i] = (int) instances.instance(i).value(instances.numAttributes() - 1);
		}// Of for i

		// Feature Selection
		int feaNum = 5;
		int[] fea = new int[feaNum];
		
		fea = new int[]{18,9,12,11,10};
		
		double[][] dataFea = new double[instances.numInstances()][feaNum];
		
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < feaNum; j++) {
				dataFea[i][j] = data[i][fea[j]-1];
			}
		}
		// End of Feature Selection
		
		
		for (double i = 0.01; i <= 0.2001; i += 0.01) {
			double tCost = 1;
			double loose = 0;
			double dc = i;
			CADU4 cadu = new CADU4(dataFea, rLabels, dc, new double[] { 4, 9 }, tCost, loose);
			cadu.computeMaxDistance();
			cadu.computeRho();
			cadu.computeRhoIndices();
			cadu.computeDelta();
			cadu.computePrio();
			cadu.computeLeaf();
			cadu.costSensitiveACLearning();
			System.out.print(cadu.totalCost() + "\t");
//			System.out.println(cadu.blockCount);
		}// Of for i
	}
	public static void main(String[] args) throws Exception {
		//testSmallData();
		//testFs();
		test1();
	}// Of main

	@Override
	public String toString() {
		StringBuilder sBuilder = new StringBuilder();
		// if (data.length >= 100) {
		// for (int i = 0; i < 10; i++) {
		// sBuilder.append(Arrays.toString(data[i]) + "\r\n");
		// }
		// sBuilder.append("...\r\n");
		// for (int i = data.length-10; i < data.length; i++) {
		// sBuilder.append(Arrays.toString(data[i]) + "\r\n");
		// }
		// }
		return "CADU [data=" + sBuilder.toString() + ",\r\n \r\n rLabel="
		// + Arrays.toString(rLabel) + ",\r\n master="
		// + Arrays.toString(master) + ",\r\n delta="
				+ Arrays.toString(delta) + ",\r\n rho=" + Arrays.toString(rho) + ",\r\n dc=" + dc + ",\r\n mCostR="
				+ mCost[0] + ",\r\n mCostB=" + mCost[1] + ",\r\n tCost=" + tCost + ",\r\n loose=" + loose + ",\r\n T="
				+ T + ",\r\n maxDistance=" + maxDistance + ",\r\n prioIndices=";
		// + Arrays.toString(prioIndices) + ",\r\n rhoIndices="
		// + Arrays.toString(rhoIndices) + ",\r\n predictedLabels="
		// + Arrays.toString(predictedLabels) + ",\r\n queried="
		// + Arrays.toString(queried) + "]";
	}// Of toString

	/**
	 ********************* 
	 * Compute the expect number of positive instances.
	 * 
	 * @param R
	 *            the number of positive instances checked.
	 * @param B
	 *            the number of negative instances checked.
	 * @param N
	 *            the total number of instances.
	 * @return the expect number of positive instances.
	 ********************* 
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
			// System.out.println("fenzi:" + fenzi + ", fenmu: " + fenmu);
		}// Of for i
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum

	/**
	 ********************* 
	 * Compute arrangement of A^m_n where m <= B
	 ********************* 
	 */
	public static BigDecimal A(int m, int n) {
		if (m > n) {
			return new BigDecimal("0");
		}// Of if
		BigDecimal re = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++) {
			re = re.multiply(new BigDecimal(i));
		}// Of if
		return re;
	}// Of A
	
	/**
	 ********************* 
	 * Quickly compute the number of arrangement by Stirling equation.
	 *********************
	 */
	public static BigDecimal quickA(int m, int n) {
		
		if (m > n) {
			return new BigDecimal(0);		
		}// Of if 
		BigDecimal nmul = new BigDecimal(Math.sqrt(2 * Math.PI * n));
		nmul = nmul.multiply(new BigDecimal(n).pow(n).divide(new BigDecimal(Math.E).pow(n), 4, BigDecimal.ROUND_HALF_EVEN));
		
		BigDecimal mmul = new BigDecimal(Math.sqrt(2 * Math.PI * (n - m)));
		if (m == n) {
			mmul = new BigDecimal(1);
		} else {
			mmul = mmul.multiply(new BigDecimal((n-m) / Math.E).pow(n-m));
		}// Of if 
		return nmul.divide(mmul, 4, BigDecimal.ROUND_HALF_EVEN);
	}// Of quickA
}// Of class CADU3
