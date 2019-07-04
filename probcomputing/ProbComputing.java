package fansactive.probcomputing;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.Arrays;

public class ProbComputing {

	/**
	 * Compute the expect number of positive instances.
	 * 
	 * @param R
	 *            the number of positive instances checked.
	 * @param B
	 *            the number of negative instances checked.
	 * @param N
	 *            the total number of instances.
	 * @return the expect number of positive instances.
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));

			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);

			// System.out.println("fenzi:" + fenzi + ", fenmu: " + fenmu);
		}
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}

	/**
	 * Compute p(R* | N, R, B)
	 * 
	 * @param R_star
	 * @param R
	 * @param B
	 * @param N
	 * @return
	 */
	public static double probOfRStar(int R_star, int R, int B, int N) {
		BigDecimal fenzi = A(R, R_star).multiply(A(B, N - R_star));
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = 0; i <= N; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));

			fenmu = fenmu.add(a);
		}
		return fenzi.divide(fenmu, 5, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}

	/**
	 * Compute \sigma(N, R, B), which could be optimization.
	 * 
	 * @param R
	 * @param B
	 * @param N
	 * @return
	 */
	public static double stddev(int R, int B, int N) {

		double re = 0;
		double expe = expectPosNum(R, B, N) / N;
		for (int i = 0; i <= N; i++) {
			re += probOfRStar(i, R, B, N) * (((double) i) / N - expe) * (((double) i) / N - expe);
		}
		return Math.sqrt(re);
	}

	/**
	 * Compute arrangement of A^m_n where m <= B
	 * 
	 * @return
	 */
	public static BigDecimal A(int m, int B) {
		if (m > B) {
			return new BigDecimal("0");
		}
		BigDecimal re = new BigDecimal("1");
		for (int i = B - m + 1; i <= B; i++) {
			re = re.multiply(new BigDecimal("" + i));
		}
		return re;
	}

	public static void test() {

		System.out.println("---------------- Unit Test ----------------");

		System.out.println("Test: R=2, B=1, N=5");
		System.out.println("r_ba: " + expectPosNum(2, 1, 5));

		System.out.println("Test: R=2, B=0, N=5");
		System.out.println("r_ba: " + expectPosNum(2, 0, 5));

		System.out.println("Test: R* = 1, R=2, B=0, N=5");
		System.out.println("p: " + probOfRStar(1, 2, 0, 5));

		System.out.println("Test: R* = 2, R=2, B=1, N=5");
		System.out.println("p: " + probOfRStar(2, 2, 1, 5));
	}

	public static void testDemo() {
//		int[] N = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
		
	}
	
	public static void main(String[] args) throws Exception{
//		test();
//		 r_ba(N, R, 0) wrt. N and R ranges from 1 to 20
//		int[] N = { 20, 40, 100, 400, 20000 };
//		for (int i = 0; i < N.length; i++) {
//			for (int j = 0; j <= 20; j++) {
//				System.out.print(expectPosNum(j, 0, N[i]) / N[i] + "\t");
//			}
//			System.out.println(";");
//		}
//		System.out.println();
//		// r_ba(N, R, 1) wrt. N and R ranges from 1 to 20
//		N = new int[] { 40, 60, 100, 400, 20000 };
//		for (int i = 0; i < N.length; i++) {
//			for (int j = 1; j <= 31; j++) {
//				System.out.print(expectPosNum(j - 1, 1, N[i]) / N[i] + "\t");
//			}
//			System.out.println(";");
//		}

		// p(R*|N, R, B) wrt. R*

//		 int R = 5;
//		 int B = 2;
//		 int N = 100;
//		 double x = 0;
//		 for (int i = 0; i <= N-B; i++) {
//			 double tx = probOfRStar(i, R, B, N);
//			 x += tx;
//			 System.out.print(tx + "\t");
//		 }
//		 System.out.println("\n" + x);
		// standard deviation.
		 
//		 System.out.println();
//		 for (int i = 0; i <= N; i++) {
//		 System.out.print(i + "\t");
//		 }
//		 System.out.println();
		
//		 for (int i = 0; i <= 100; i++) {
//			 System.out.printf("%.4f\t", stddev(i, 0, 100));
//		
//		 }
		//
		// System.out.println(stddev(0, 0, 100));
		//
		// System.out.println(stddev(0, 2, 100));

//		 for (int i = 2; i <= 50 ; i++) {
//		 System.out.print(expectPosNum(i, 0, 100)/100 + "\t");
//		 }
//		 
//		 System.out.println();
//		
//		 for (int i = 2; i <= 50; i++) {
//		 System.out.print(expectPosNum(2*i-1, 1, 100)/100 + "\t");
//		 }
//		 System.out.println();
//		 for (int i = 2; i <= 50; i++) {
//		 System.out.print(expectPosNum(2*i-2, 2, 100)/100 + "\t");
//		 }

		// Look up table for r_ba(N, R, 0)
		// int N = 5;
		//
		// for (int i = 0; i <= 2; i++) {
		// double x = expectPosNum(i, 0, N)/N;
		// double dev = stddev(i, 0, N);
		//
		// System.out.print(i + " " + x + " " + dev + "\n");
		// }
		// System.out.println("\t");


		double m01 = 4;
		double t = 1;
		int[] minValueIndex = new int[500];
		double[][] fValues = new double[500][101];
		
//		for (int i = 0; i <= 10; i++) {
//			System.out.print(expectPosNum(i, 0, 10)+"\t");
//		}
		System.out.println();
		for (int N = 100; N <= 100; N++) {
//			System.out.println(N);
			Arrays.fill(fValues[N-1], Double.MAX_VALUE);
			double minValue = 0.5 * N * m01;
			fValues[N-1][0] = minValue;
			System.out.printf("%.4f\t", fValues[N-1][0]);
			int minIndex = 0;
			for (int i = 1; i <= N; i++) {
				double fvalue = (1 - expectPosNum(i, 0, N) / N) * N * m01 + t * i;
				if (fvalue < minValue) {
					minIndex = i;
					minValue = fvalue;
				}
				System.out.printf("%.4f\t", fvalue);
				fValues[N-1][i] = fvalue;
			}
//			System.out.println(minValue);
			
			minValueIndex[N-1] = minIndex;
//			System.out.println("],...");
		}
//		System.out.println(Arrays.toString(fValues[99]));
//		System.out.println(Arrays.toString(minValueIndex));
//		Utils.writeMatrix2DToMat(new File("src/data/mat/test.mat"), Matrix.Factory.importFromArray(fValues), "X");
		
		// int[][] minRIndex = { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 5, 3 }, { 6, 8,
		// 4 }, { 9, 11, 5 }, { 12, 16, 6 },
		// { 17, 20, 7 }, { 21, 25, 8 }, { 26, 30, 9 }, { 31, 36, 10 }, { 37,
		// 43, 11 }, { 44, 50, 12 },
		// { 51, 58, 13 }, { 59, 66, 14 }, { 67, 74, 15 }, { 75, 83, 16 }, { 84,
		// 93, 17 }, { 94, 103, 18 },
		// { 104, 113, 19 }, { 114, 124, 20 }, { 125, 136, 21 }, { 137, 147, 22
		// }, { 148, 160, 23 },
		// { 161, 173, 24 }, { 174, 186, 25 }, { 187, 200, 26 }, { 201, 215, 27
		// }, { 216, 230, 28 },
		// { 231, 246, 29 }, { 247, 261, 30 }, { 262, 278, 31 }, { 279, 295, 32
		// }, { 296, 313, 33 },
		// { 314, 330, 34 }, { 331, 349, 35 }, { 350, 368, 36 }, { 369, 388, 37
		// }, { 389, 408, 38 },
		// { 409, 428, 39 }, { 429, 449, 40 }, { 450, 471, 41 }, { 472, 492, 42
		// }, { 493, 500, 43 }, };
		// int count = 0;
		// int integerTimes = 0;
		// for (int i = 1; i <= minRIndex.length; i++) {
		// System.out.print(" & " + minRIndex[i - 1][0] + "\\textasciitilde " +
		// minRIndex[i - 1][1]);
		// if (i % 9 == 0) {
		// System.out.print("\\\\\t\n$N$");
		// }
		// }
		// System.out.println();
		// for (int i = 1; i <= minRIndex.length; i++) {
		// System.out.print(" & " + minRIndex[i-1][2]);
		// if (i % 9 == 0) {
		// System.out.print("\\\\\t\n$R^*$");
		// }
		// }
	}
}
