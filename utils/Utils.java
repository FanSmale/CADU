package fansactive.utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import javax.rmi.CORBA.Util;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import fansactive.probcomputing.ProbComputing;
import weka.core.Instance;
import weka.core.Instances;

public class Utils {
	
	public static Instances randomSelect(Instances instances, double ratio) {
		
		
				
		for (int i = 0; i < instances.numInstances(); i++) {
			
		}
		
		return null;
	}
	
	public static void writeMatrix2DToMat(File f, Matrix matrix, String whichMatrix) throws IOException {
		org.ujmp.jmatio.ExportMatrixMAT.toFile(f, matrix, whichMatrix);
	}

	public static Eig[] eig(double[][] matrix) {

		Matrix matrix2 = Matrix.Factory.importFromArray(matrix);
		
		Matrix[] reMatrixs = matrix2.eig();

		Eig[] re = new Eig[matrix.length];

		for (int i = 0; i < re.length; i++) {
			Matrix vec = reMatrixs[0].selectColumns(Ret.NEW, i).transpose();

			re[i] = new Eig(reMatrixs[1].getAsDouble(i, i), vec.toDoubleArray()[0]);
		}

		return re;
	}

	public static class Eig {
		double eigValue;
		double[] eigVector;

		public Eig(double eigValue, double[] eigVector) {
			this.eigValue = eigValue;
			this.eigVector = eigVector;
		}

		@Override
		public String toString() {
			return "Eig [eigValue=" + eigValue + ", eigVector=" + Arrays.toString(eigVector) + "]";
		}

	}

	public static void goldSearch(double start, double end, double eps) {

		double a, b;
		b = Math.max(start, end);
		a = Math.min(start, end);

		double t1, t2, f1, f2;
		double ranta = 0.618034f;

		t1 = b - ranta * (b - a);
		t2 = a + ranta * (b - a);
		f1 = Math.sin(t1);
		f2 = Math.sin(t2);

		while (t2 - t1 > eps) {

			if (f1 <= f2)
				b = t2;
			else
				a = t1;
			t1 = b - ranta * (b - a);
			t2 = a + ranta * (b - a);
			f1 = Math.sin(t1);
			f2 = Math.sin(t2);
		}

		double x, y;
		if (f1 > f2) {
			x = t2;
			y = f2;
		} else {
			x = t1;
			y = f1;
		}
		System.out.println(x + "\t" + y);
	}

	public static double[] goldSearch1(int N, double mCost, double tCost, double eps) {

		int start = 0;
		int end = N;

		int a, b;
		b = Math.max(start, end);
		a = Math.min(start, end);
		
		int t1, t2;
		double f1, f2;
		double ranta = 0.618034f;
		
		t1 = start;
		t2 = end;
		f1 = fun(t1, N, mCost, tCost);
		f2 = fun(t2, N, mCost, tCost);
		
		while (Math.abs(t1 - t2) > 1) {

			if (f1 <= f2)
				b = t2;
			else
				a = t1;
			t1 = (int) (b - ranta * (b - a));
			t2 = (int) (a + ranta * (b - a));
			f1 = fun(t1, N, mCost, tCost);
			f2 = fun(t2, N, mCost, tCost);
		}

		int x;
		double y;
		if (f1 > f2) {
			x = t2;
			y = f2;
		} else {
			x = t1;
			y = f1;
		}
		System.out.println(x);

		int reSearchLength = 2;
		int minIndex = x;
		double minValue = y;
		for (int i = -reSearchLength; i <= reSearchLength; i++) {
			if (x + i < 0 || x + i >= N || x == 0) {
				continue;
			}
			double t = fun(x + i, N, mCost, tCost);
			if (t < minValue) {
				minValue = t;
				minIndex = x+i;
			}
		}
		return new double[]{minIndex, minValue};
	}

	public static double fun(int R, int N, double mCost, double tCost) {
		double r_ba = ProbComputing.expectPosNum(R, 0, N) / N;
		return mCost * (1 - r_ba) * N + tCost * R;
	}

	public static void main(String[] args) throws Exception {
		// Matrix xMatrix = Matrix.Factory.rand(5, 5);
		// System.out.println(Arrays.toString(xMatrix.eig()));
		//
		// Eig[] rEigs = eig(xMatrix.toDoubleArray());
		// for (int i = 0; i < rEigs.length; i++) {
		// System.out.println(rEigs[i]);
		// }
//		for (int i = 1; i <= 1000; i++) {
//			goldSearch1(i, 2, 1, 1);
//		}
		String[] dString = {"flare-solar", "german", "heart", "image", "splice", "titanic"};
		for (int i = 0; i < dString.length; i++) {
			Matrix X = Utils1.loadMatrix2DFromMat(new File("src/data/mat/"+dString[i]+"_test_5000.mat"), "X");
			Matrix Y = Utils1.loadMatrix2DFromMat(new File("src/data/mat/"+dString[i]+"_test_5000.mat"), "Y");
			
			Utils1.writeMatrixToArffFileWithClassMessage(X, Y, "src/data/arff/"+dString[i]+"_test_5000.arff", dString[i]);
		}// Of for i
	}
}