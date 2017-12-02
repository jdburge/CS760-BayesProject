package hab.cs760;

import com.sun.istack.internal.Nullable;
import hab.cs760.bayesnet.BayesNet;
import hab.cs760.bayesnet.Util;
import hab.cs760.machinelearning.ArffReader;
import hab.cs760.machinelearning.Feature;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BayesLearner {
	private final boolean isNaive;
	private final List<Feature> featureList;
	public final List<Instance> trainInstances;
	public BayesNet net;
	private final List<Instance> testInstances;

	public BayesLearner(String trainFile, String testFile, boolean isNaive) {
		ArffReader arffReader = readFile(trainFile);
		featureList = arffReader.getFeatureList();
		trainInstances = arffReader.getInstances();

		arffReader = readFile(testFile);
		testInstances = arffReader.getInstances();

		this.isNaive = isNaive;
	}

	private BayesLearner(List<Feature> featureList, List<Instance> trainInstances, List<Instance>
			testInstances, boolean isNaive) {
		this.featureList = featureList;
		this.trainInstances = trainInstances;
		this.testInstances = testInstances;
		this.isNaive = isNaive;
	}

	public static void main(String[] args) {
		if (args.length == 2) {
			if (args[0].equals("chess")) {
				runChessAnalysis(args[1]);
				System.exit(1);
			}
		}
		if (args.length < 3) {
			printUsage();
			System.exit(1);
		}
		learnBayesAndPrintResults(args);
	}

	private static void runChessAnalysis(String dataFile) {
		ArffReader arffReader = readFile(dataFile);
		List<Feature> featureList = arffReader.getFeatureList();
		List<Instance> instances = arffReader.getInstances();
		List<List<Instance>> subsets = Util.subsetsUsingStratifiedSampling(instances, 10);

		System.out.println("naive\ttan");
		for (List<Instance> testInstances : subsets) {
			// make training set
			List<Instance> trainInstances = new ArrayList<>();
			for (List<Instance> subset : subsets) {
				if (subset != testInstances) trainInstances.addAll(subset);
			}

			// naive bayes
			BayesLearner bayesLearner = new BayesLearner(featureList, trainInstances,
					testInstances, true);
			bayesLearner.buildBayes();
			System.out.print(String.format("%.6f", bayesLearner.getAccuracyOnTestset()));
			System.out.print("\t");

			// tan bayes
			bayesLearner = new BayesLearner(featureList, trainInstances, testInstances, false);
			bayesLearner.buildBayes();
			System.out.print(String.format("%.6f", bayesLearner.getAccuracyOnTestset()));

			System.out.println();
		}
	}

	private static void learnBayesAndPrintResults(String[] args) {
		String mode = args[2];
		boolean isNaive = false;
		if (mode.equals("n") || mode.equals("t")) {
			isNaive = mode.equals("n");
		} else {
			errorExitWithMessage("<n|t> argument is malformed. Should be either n or t");
		}

		BayesLearner learner = new BayesLearner(args[0], args[1], isNaive);
		learner.buildBayes();
		System.out.println(learner.bayesNetString());
		System.out.println();
		System.out.println(learner.testSetPredictions());
	}

	private static void errorExitWithMessage(String message) {
		System.out.println(message);
		System.out.println();
		printUsage();
		System.exit(1);
	}

	private static void printUsage() {
		System.out.println("Usage:  bayes <train-set-file> <test-set-file> <n|t>\n use n for " +
				"Naive Bayes, t for TAN");
		System.out.println("To run chess analysis:\n "
				+ "bayes chess <chess-file>\n");
	}

	/**
	 * @param fileName name of the file that should be read
	 * @return null if there was an exception when trying to read the file. otherwise, arff
	 * reader object with file contents
	 */
	@Nullable
	private static ArffReader readFile(String fileName) {
		try {
			return new ArffReader(fileName);
		} catch (FileNotFoundException e) {
			errorExitWithMessage("File with name " + fileName + " not found: " + e.getMessage());
		} catch (IOException e) {
			errorExitWithMessage("IOException: " + e.getMessage());
		}
		return null;
	}

	public String testSetPredictions() {
		return testSetPredictions(true);
	}

	private String testSetPredictions(boolean showIndividualPredictions) {
		StringBuilder builder = new StringBuilder();
		NominalFeature classLabel = (NominalFeature) featureList.get(featureList.size() - 1);

		int correctCount = 0;
		for (Instance instance : testInstances) {
			String actualLabel = instance.actualLabel;
			String predictedLabel;
			double prediction = net.predictProbabilityOfFirstLabelValue(instance, isNaive);
			if (prediction < 0.5) {
				predictedLabel = classLabel.possibleValues.get(1);
				prediction = 1 - prediction;
			} else {
				predictedLabel = classLabel.possibleValues.get(0);
			}
			if (actualLabel.equals(predictedLabel)) correctCount++;
			if (showIndividualPredictions) {
				if (builder.length() > 0) builder.append(("\n"));
				builder.append(String.format("%s %s %.12f", predictedLabel, actualLabel,
						prediction));

			}
		}
		if (showIndividualPredictions) {
			builder.append("\n\n");
		}
		builder.append(correctCount);

		return builder.toString();
	}

	private double getAccuracyOnTestset() {
		return (double) Integer.parseInt(testSetPredictions(false)) / testInstances.size();
	}

	public void buildBayes() {
		if (isNaive) {
			net = BayesNet.naiveNet(featureList);
			net.train(trainInstances);
		} else {
			net = BayesNet.treeAugmentedNet(featureList, trainInstances);
			net.train(trainInstances);
			net.makeFair();
		}
	}

	public String bayesNetString() {
		return net.toString();
	}

}
