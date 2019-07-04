package fansactive.clustering;

import java.io.FileReader;
import java.util.Arrays;

import fansactive.utils.Utils1;
import weka.core.Instances;

public class TSDCADU {

	private TwoRoundMeans twoRoundMeans;
	
	private CADU4 cadu;
	
	double[][] data;
	
	double dc;
	
	int[] predictedLabels;
	
	double[] rho;
	
	double[] delta;
	
	int[] master;
	
	double maxDistance;
	
	int[] rhoIndices;
	
	int[] prioIndices;
	
	boolean[] queried;
	
	
	boolean[] leaf;
	
	
	public TSDCADU(double[][] pData, int[] pRLabels, double pDc, double[] pMCost, double pTCost, double pLoose) {
		data = pData;
		twoRoundMeans = new TwoRoundMeans(pData);
		twoRoundMeans.cluster();
		cadu = new CADU4(twoRoundMeans.virtualCenters, pRLabels, pDc, pMCost, pTCost, pLoose);
	}
	
	public void activeLearning() {
		
		rho = twoRoundMeans.rho;
		cadu.computeMaxDistance();
		maxDistance = cadu.maxDistance;
		cadu.computeRhoIndices();
		rhoIndices = cadu.rhoIndices;
		cadu.computeDelta();
		delta = cadu.delta;
		master = cadu.master;
		cadu.computePrio();
		prioIndices = cadu.prioIndices;
		cadu.computeLeaf();
		leaf = cadu.leaf;
		
		
		
		
	}// Of activeLearning
	
	public double totalCost() {
		
		return 0;
	}
	
	public static class TwoRoundMeans {

		/**
		 * The data matrix. The number of row is the number of instances and the
		 * number of column is the number of features.
		 */
		double[][] data;

		/**
		 * The number of clusters, default is int(sqrt(n)).
		 */
		int numCluster;

		/**
		 * Virtual centers. virtualCenters[i] represents the (i-1)-th cluster.
		 */
		double[][] virtualCenters;

		/**
		 * Predicted clusters labels. Ranges from [0, int(sqrt(n))-1].
		 */
		int[] preClusters;

		int[][] blocks;
		
		/**
		 * The rho for each blocks.
		 */
		double[] rho;

		public TwoRoundMeans(double[][] pData) {
			data = pData;
			numCluster = (int) Math.sqrt(data.length);
		}// Of TwoRoundMeans

		public TwoRoundMeans(double[][] pData, int pNumClusters) {
			data = pData;
			numCluster = pNumClusters;
		}// Of TwoRoundMeans

		public void cluster() {

			int n = data.length;

			int[] tCenterIndices = Arrays.copyOfRange(Utils1.randomPermutationArray(0, n), 0, numCluster);
			
			double[][] tVirtualCenters = new double[numCluster][];
			
			for (int i = 0; i < tCenterIndices.length; i++) {
				tVirtualCenters[i] = Arrays.copyOf(data[tCenterIndices[i]], data[tCenterIndices[i]].length);
			}// Of for i
			
			// 2-Round-Means
			for (int i = 0; i < 2; i++) {
				
				int[] tCluster = new int[data.length];
				for (int j = 0; j < n; j++) {
					double tMinDist = Double.MAX_VALUE;
					int tMinDistIndex = 0;
					for (int k = 0; k < numCluster; k++) {
						double tDist = euclideanDist(data[j], tVirtualCenters[k]);
						if (tDist < tMinDist) {
							tDist = tMinDist;
							tMinDistIndex = k;
						}// Of if
					}// Of for k
					tCluster[j] = tMinDistIndex;
				}// Of for j
				
				// Rebuild centers
				int[] count = new int[numCluster];
				for (int j = 0; j < data.length; j++) {
					count[tCluster[j]]++;
				}// Of for j
				
				// Blocks
				int[][] tBlocks = new int[numCluster][];
				for (int j = 0; j < count.length; j++) {
					tBlocks[j] = new int[count[j]];
					count[j] = 0;
				}// Of for j
				
				for (int j = 0; j < data.length; j++) {
					tBlocks[tCluster[j]][count[tCluster[j]]++] = j;
				}// Of for j
				
				// Find new clusters
				
				for (int j = 0; j < tBlocks.length; j++) {
					Arrays.fill(tVirtualCenters[j], 0);
					for (int k = 0; k < data[0].length; k++) {
						for (int m = 0; m < tBlocks[j].length; m++) {
							tVirtualCenters[j][k] += data[tBlocks[j][m]][k];
						}// Of for m
						tVirtualCenters[j][k] /= tBlocks[j].length;
					}// Of for k
				}// Of for j
				
				preClusters = tCluster;
				blocks = tBlocks;
			}// Of for i
			
			virtualCenters = tVirtualCenters;
		}// Of cluster
		
		public double euclideanDist(double[] a, double[] b) {
			double sum = 0;
			for (int i = 0; i < a.length; i++) {
				sum += (a[i] - b[i]) * (a[i] - b[i]);
			}// Of for i
			return Math.sqrt(sum);
		}// Of euclideanDist
		
		public double mahattenDist(double[] a, double[] b) {
			double sum = 0;
			for (int i = 0; i < a.length; i++) {
				sum += Math.abs(a[i] - b[i]);
			}// Of for i
			return sum;
		}// Of mahattenDist
	}// Of class TwoRoundMeans
	
	public static void test() throws Exception {
		// Instances instances = new Instances(new
		// FileReader("src/data/iris.arff"));
		// Instances instances = new Instances(new FileReader(
		// "src/data/breast-cancer_test.arff"));
		double dc = 0.05;
		for (; dc < 0.601; dc += 0.05) {
			Instances instances = new Instances(new FileReader("src/data/arff/thyroid_test.arff"));
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

			double loose = 0.2;

			int T = 3;

			TSDCADU cadu = new TSDCADU(data, rLabels, dc, new double[] { 4, 9 }, tCost, loose);
			
			// CADU cadu2 = new CADU(data, rLabels, dc, 4, 9, tCost, T, loose);
			cadu.activeLearning();

			// cadu.computeMaxDistance();
			// cadu.computeRho();

			// System.out.println(Arrays.toString(cadu.rho));

			// cadu2.costSensitiveACLearning();

			// System.out.println(Arrays.equals(cadu.predictedLabels,
			// cadu2.predictedLabels));

//			System.out.println(Arrays.toString(cadu.blockInfo));
//			System.out.println("Total cost:");
//			System.out.println(cadu.totalCost());
	//
//			System.out.println(Arrays.toString(cadu.master));
			
//			System.out.print(cadu.maxsizePureBlock + "\t");
		}// Of for dc
	}// Of test

	
	public static void main(String[] args) throws Exception {
		test();
	}
}// Of class TSDCADU
