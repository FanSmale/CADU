package fansactive.clustering;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Map;

import fansactive.probcomputing.ProbComputing;
import weka.core.Instances;

public class CADU {

	double[][] data;

	double[][] distance;

	int[] rLabel;

	int[] master;

	double[] delta;

	double[] rho;

	double dc;

	double mCostR;

	double mCostB;

	double tCost;

	double loose;

	/**
	 * The minimum threshold to queried all instances.
	 */
	double T;

	double maxDistance;

	public CADU(double[][] pData, int[] pRLabels, double pDc, double pMCostR, double pMCostB, double pTCost,
			int pT, double pLoose) {
		rLabel = pRLabels;
		data = pData;
		dc = pDc;
		mCostR = pMCostR;
		mCostB = pMCostB;
		loose = pLoose;
		T= pT;
		tCost = pTCost;
	}

	public void costSensitiveACLearning() {

		computeDistance();
		computeMaxDistance();
		computeRho();
		computeRhoIndices();
		computeDelta();
		computePrio();

		queried = new boolean[data.length];
		predictedLabels = new int[data.length];
		Arrays.fill(predictedLabels, -1);

		int[] pIndices = new int[data.length];
		for (int i = 0; i < pIndices.length; i++) {
			pIndices[i] = i;
		}
		// blockLearning(pIndices);
		blockLearning(rhoIndices);
	}

	int[] prioIndices;
	int[] rhoIndices;
	int[] predictedLabels;
	boolean[] queried;
	int blockCount = 0;
	public void blockLearning(int[] pIndices) {
		blockCount++;
		// 1. Stop criterion
		// 1.1 All are labeled
		boolean allLabeled = true;
		for (int i = 0; i < pIndices.length; i++) {
			if (predictedLabels[pIndices[i]] == -1) {
				allLabeled = false;
			}
		}
		if (allLabeled) {
			return;
		}

		// 1.2 This block is too small and queried all
		if (pIndices.length <= T) {
			for (int i = 0; i < pIndices.length; i++) {
				if (predictedLabels[pIndices[i]] == -1) {
					queried[pIndices[i]] = true;
					predictedLabels[pIndices[i]] = rLabel[pIndices[i]];
				}
			}
			return;
		}

		// cPrioInd = [a1, ..., am] where rho*delta[a1] >= ... >= rho*delta[am]
		int[] cPrioInd = new int[pIndices.length];
		{
			double[] tmpPrio = new double[pIndices.length];

			for (int i = 0; i < tmpPrio.length; i++) {
				tmpPrio[i] = rho[pIndices[i]] * delta[pIndices[i]];
			}
			int[] tmpPrioIndices = argSort(tmpPrio, Order.DESC);
			for (int i = 0; i < cPrioInd.length; i++) {
				cPrioInd[i] = pIndices[tmpPrioIndices[i]];
			}
		}
//		System.out.println(Arrays.toString(cPrioInd));
		// 1.3 If this block is not pure.
		boolean[] notPure = new boolean[2];
		for (int i = 1; i < pIndices.length; i++) {
			if (predictedLabels[pIndices[i]] == 0) {
				notPure[0] = true;
			}
			if (predictedLabels[pIndices[i]] == 1) {
				notPure[1] = true;
			}
			// Split
			if (notPure[0] && notPure[1]) {
				int[][] p = splitTwoBlocks(pIndices, new int[] { cPrioInd[0], cPrioInd[1] });
				blockLearning(p[0]);
				blockLearning(p[1]);
				return;
			}
		}
		// 2. Query a part of this block
		// 2.1 // Look up
//		System.out.println(pIndices.length);
		int[] RandB = lookup(pIndices.length);
//		System.out.println(Arrays.toString(RandB));
		///////
		int R = RandB[0];
		int B = RandB[1];
		boolean split = false;

		// 2.2. Query instances
		// There are some cases: 
		// 2.2.1. Number of already queried is less than min(R, B)
		// 2.2.2. Number of already queried is greater than min(R, B) and less than max(R, B)
		// 2.2.3. Number of already queried is greater than max(R, B)
		// We consider all these cases.
		if (R <= B) {
			for (int i = 0; i < R; i++) {
				queried[cPrioInd[i]] = true;
				predictedLabels[cPrioInd[i]] = rLabel[cPrioInd[i]];
				if (predictedLabels[cPrioInd[i]] != predictedLabels[cPrioInd[0]]) {
					split = true;
					break;
				}
			}
			if (!split && predictedLabels[cPrioInd[0]] == 1) {
				for (int i = R; i < B; i++) {
					queried[cPrioInd[i]] = true;
					predictedLabels[cPrioInd[i]] = rLabel[cPrioInd[i]];
					if (predictedLabels[cPrioInd[i]] != predictedLabels[cPrioInd[0]]) {
						split = true;
						break;
					}
				}
			}
		} else {
			for (int i = 0; i < B; i++) {
				queried[cPrioInd[i]] = true;
				predictedLabels[cPrioInd[i]] = rLabel[cPrioInd[i]];
				if (predictedLabels[cPrioInd[i]] != predictedLabels[cPrioInd[0]]) {
					split = true;
					break;
				}
			}
			if (!split && predictedLabels[cPrioInd[0]] == 0) {
				for (int i = B; i < R; i++) {
					queried[cPrioInd[i]] = true;
					predictedLabels[cPrioInd[i]] = rLabel[cPrioInd[i]];
					if (predictedLabels[cPrioInd[i]] != predictedLabels[cPrioInd[0]]) {
						split = true;
						break;
					}
				}
			}
		}
		
		// Predict
		if (!split) {
			for (int i = 0; i < cPrioInd.length; i++) {
				if (predictedLabels[cPrioInd[i]] == -1) {
					predictedLabels[cPrioInd[i]] = predictedLabels[cPrioInd[0]];
				}
			}
			return;
		}

		// Split
		int[][] p = splitTwoBlocks(pIndices, new int[] { cPrioInd[0], cPrioInd[1] });
		blockLearning(p[0]);
		blockLearning(p[1]);
		return;
	}

	private int[] lookup(int pSize) {
		// Linear Search
		double tmpRMinCost = 0.5 * mCostR * pSize;
		double tmpBMinCost = 0.5 * mCostB * pSize;
		int Rstar = 0;
		int Bstar = 0;
		
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		ra[0] = 0.5;
		for (int i = 1; i <= pSize; i++) {
			ra[i] = ProbComputing.expectPosNum(i, 0, pSize) / pSize;
			double tmpRCost = (1 - ra[i]) * mCostR * pSize + tCost * i;
			double tmpBCost = (1 - ra[i]) * mCostB * pSize + tCost * i;
			if (tmpRCost < tmpRMinCost) {
				tmpRMinCost = tmpRCost;
				Rstar = i;
			} else {
				isFind[0] = true;
			}
			if (tmpBCost < tmpBMinCost) {
				tmpBMinCost = tmpBCost;
				Bstar = i;
			} else {
				isFind[1] = true;
			}
			if (isFind[0] && isFind[1]) {
				ra = Arrays.copyOf(ra, i);
				break;
			}
		}
		
		int RstarLoose = 0;
		double costRloose = 0;
		int BstarLoose = 0;
		double costBloose = 0;
		
		Arrays.fill(isFind, false);
		
		for (int i = 0; i < ra.length; i++) {
			double tLooseCost = (1 - ra[i]) * mCostR * pSize + tCost * i;
			if (tLooseCost <= (1+loose) * tmpRMinCost && !isFind[0]) {
				isFind[0] = true;
				RstarLoose = i;
				costRloose = tLooseCost;
			}
			
			tLooseCost = (1 - ra[i]) * mCostB * pSize + tCost * i;
			if (tLooseCost <= (1 + loose) * tmpBMinCost && !isFind[1]) {
				isFind[1] = true;
				BstarLoose = i;
				costBloose = tLooseCost;
			}
			
			if (isFind[0] && isFind[1]) {
				break;
			}
		}
		return new int[]{RstarLoose, BstarLoose};
	}
	
	/**
	 * Split a impure block to two sub blocks.
	 * 
	 * @param sortedBlock
	 *            a block of indices sorted by rho.
	 * @param centers
	 *            center indices.
	 * @return
	 */
	private int[][] splitTwoBlocks(int[] sortedBlock, int[] centers) {
		int[][] twoBlocks = new int[2][sortedBlock.length];

		int[] cl = new int[master.length];
		Arrays.fill(cl, -1);

		cl[centers[0]] = 0;
		cl[centers[1]] = 1;
		int[] count = new int[2];
		for (int i = 0; i < sortedBlock.length; i++) {
			if (cl[sortedBlock[i]] == -1) {
				cl[sortedBlock[i]] = cl[master[sortedBlock[i]]];
				twoBlocks[cl[sortedBlock[i]]][count[cl[sortedBlock[i]]]] = sortedBlock[i];
				count[cl[sortedBlock[i]]]++;
			} else if (sortedBlock[i] == centers[0]) {
				twoBlocks[0][count[0]] = sortedBlock[i];
				count[0]++;
			} else if (sortedBlock[i] == centers[1]) {
				twoBlocks[1][count[1]] = sortedBlock[i];
				count[1]++;
			}
		}
		int[][] re = new int[2][];
		for (int i = 0; i < 2; i++) {
			re[i] = Arrays.copyOf(twoBlocks[i], count[i]);
		}
		return re;
	}

	/**
	 * Euclidean Distance
	 */
	public void computeDistance() {
		double[][] tDistance = new double[data.length][data.length];

		for (int i = 0; i < data.length; i++) {
			for (int j = i + 1; j < data.length; j++) {
				double t = 0;
				for (int k = 0; k < data[0].length; k++) {
					//t += (data[i][k] - data[j][k]) * (data[i][k] - data[j][k]);
					t += Math.abs(data[i][k] - data[j][k]);
				}
				tDistance[i][j] = t;
				tDistance[j][i] = t;
			}
		}
		distance = tDistance;
	}

	public void computeMaxDistance() {
		double tMaxDistance = Double.MIN_VALUE;
		for (int i = 0; i < data.length; i++) {
			for (int j = i + 1; j < data.length; j++) {
				if (tMaxDistance < distance[i][j]) {
					tMaxDistance = distance[i][j];
				}
			}
		}
		maxDistance = tMaxDistance;
	}

	public void computeRho() {
		double[] tRho = new double[data.length];
		double dcDist = dc * maxDistance;

		for (int i = 0; i < data.length; i++) {
			for (int j = i + 1; j < data.length; j++) {
				if (distance[i][j] < dcDist) {
					tRho[i]++;
					tRho[j]++;
				}
			}
		}
		rho = tRho;
	}

	public void computeRhoIndices() {
		rhoIndices = argSort(rho, Order.DESC);
	}

	public double totalCost() {
		double sum = 0;
		for (int i = 0; i < data.length; i++) {
			if (queried[i]) {
				sum += tCost;
			} else {
				//Actual positive; Predict negative
				if (predictedLabels[i] == 0 && rLabel[i] == 1) {
					sum += mCostR;
				} else if (predictedLabels[i] == 1 && rLabel[i] == 0) {
					sum += mCostB;
				}
			}
		}
		return sum;
	}
	
	public void computeDelta() {
		double[] tDelta = new double[data.length];
		int[] tMaster = new int[data.length];
		Arrays.fill(tMaster, -1);
		Arrays.fill(tDelta, maxDistance);

		for (int i = 1; i < data.length; i++) {
			double tMinDist = maxDistance;
			int tMinIndex = 0;
			for (int j = 0; j < i; j++) {
				if (distance[rhoIndices[i]][rhoIndices[j]] < tMinDist) {
					tMinDist = distance[rhoIndices[i]][rhoIndices[j]];
					tMinIndex = j;
				}
			}
			tDelta[rhoIndices[i]] = distance[rhoIndices[i]][rhoIndices[tMinIndex]];
			tMaster[rhoIndices[i]] = rhoIndices[tMinIndex];
		}
		delta = tDelta;
		master = tMaster;
	}

	public void computePrio() {
		double[] tmpProduct = new double[data.length];
		for (int i = 0; i < tmpProduct.length; i++) {
			tmpProduct[i] = rho[i] * delta[i];
		}
		prioIndices = argSort(tmpProduct, Order.DESC);
	}

	// Sorting order
	public enum Order {
		ASC, DESC
	}

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
		}
		// lambda function
		Arrays.sort(entries, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
		int[] indices = new int[valueArray.length];
		for (int i = 0; i < valueArray.length; i++) {
			indices[i] = entries[i].getKey();
		}
		return indices;
	}

	public static class MyEntry<K, V> implements Map.Entry<K, V> {
		K key;
		V value;

		public MyEntry(K key, V value) {
			super();
			this.key = key;
			this.value = value;
		}

		@Override
		public V getValue() {
			return value;
		}

		@Override
		public V setValue(V value) {
			V v = this.value;
			this.value = value;
			return v;
		}

		@Override
		public K getKey() {
			return key;
		}

		@Override
		public String toString() {
			return "(" + key + ", " + value + ")";
		}
	}

	public static void test() throws Exception {
		Instances instances = new Instances(new FileReader("src/data/arff/colon.arff"));
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
		}

		// Actual positive; Predict negative
		double mCostR = 4;

		// Actual negative; Predict positive
		double mCostB = 9;

		double tCost = 1;

		double loose = 0.05;

		int T = 3;

		double dc = 0.5;

		CADU cadu = new CADU(data, rLabels, dc, mCostR, mCostB, tCost, T, loose);
		System.out.println(Arrays.toString(cadu.lookup(4)));
		cadu.costSensitiveACLearning();
		
		System.out.println(cadu);
		System.out.println("Total cost:");
		System.out.println(cadu.totalCost());
	}

//	@Override
//	public String toString() {
//		StringBuilder sBuilder= new StringBuilder();
//		sBuilder.append("data: \r\n" + Arrays.deepToString(data));
//		sBuilder.append("\r\nrealLabels: \r\n" + Arrays.toString(rLabel));
//		sBuilder.append("\r\nmaster: \r\n" + Arrays.toString(master));
//		sBuilder.append("\r\nrho: \r\n" + Arrays.toString(rho));
//		sBuilder.append("\r\ndelta: \r\n" + Arrays.toString(delta));
//		sBuilder.append("\r\npredictedLabels: \r\n" + Arrays.toString(predictedLabels));
//		
//		return sBuilder.toString();
//	}
	

	public static void main(String[] args) throws Exception {
		test();

	}

	@Override
	public String toString() {
		StringBuilder sBuilder = new StringBuilder();
//		if (data.length >= 100) {
//			for (int i = 0; i < 10; i++) {
//				sBuilder.append(Arrays.toString(data[i]) + "\r\n");
//			}
//			sBuilder.append("...\r\n");
//			for (int i = data.length-10; i < data.length; i++) {
//				sBuilder.append(Arrays.toString(data[i]) + "\r\n");
//			}
//		}
		return "CADU [data=" + sBuilder.toString() + ",\r\n \r\n rLabel="
				+ Arrays.toString(rLabel) + ",\r\n master=" + Arrays.toString(master) + ",\r\n delta=" + Arrays.toString(delta)
				+ ",\r\n rho=" + Arrays.toString(rho) + ",\r\n dc=" + dc + ",\r\n mCostR=" + mCostR + ",\r\n mCostB=" + mCostB
				+ ",\r\n tCost=" + tCost + ",\r\n loose=" + loose + ",\r\n T=" + T + ",\r\n maxDistance=" + maxDistance
				+ ",\r\n prioIndices=" + Arrays.toString(prioIndices) + ",\r\n rhoIndices=" + Arrays.toString(rhoIndices)
				+ ",\r\n predictedLabels=" + Arrays.toString(predictedLabels) + ",\r\n queried=" + Arrays.toString(queried)
				+ "]";
	}
}
