package hab.cs760.bayesnet;

import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hannah on 11/6/17.
 */
public class Util {
	/**
	 * Calculates the conditional mutual information between two features x₁, x₂ with respect to
	 * the label y, over a set of instances.
	 * I(X₁, X₂ | Y) = ∑(x₁∈values(X₁)) ∑(x₂∈values(X₂)) ∑(y∈values(Y)) P(x₁, x₂, y) log₂(P(x₁, x₂ | y) / (P(x₁ | y) P(x₂ | y))
	 * @param x1 first feature
	 * @param x2 second feature
	 * @param classLabel class label
	 * @param instances set of instances to consider
	 * @return
	 */
	public static double conditionalMutualInformation(NominalFeature x1, NominalFeature x2,
													  NominalFeature
			classLabel,
													  List<Instance> instances) {
		if (x1 == x2) return -1;
		double sum = 0;
		for (String y : classLabel.possibleValues) {
			for (String possibleX2Value : x2.possibleValues) {
				for (String possibleX1Value : x1.possibleValues) {

					InstanceCounter.Criterion x1Criterion = new InstanceCounter.Criterion(x1, possibleX1Value);
					InstanceCounter.Criterion x2Criterion = new InstanceCounter.Criterion(x2, possibleX2Value);
					InstanceCounter.Criterion classCriterion = new InstanceCounter.Criterion
							(classLabel, y);

					double Px1Giveny = InstanceCounter.probabilityOfCriteriaGivenLabel
							(classLabel, y, instances, x1Criterion);
					double Px2Giveny = InstanceCounter.probabilityOfCriteriaGivenLabel
							(classLabel, y, instances, x2Criterion);

					double Px1x2y = InstanceCounter.probabilityOfCriteriaGivenLabel(null,
							null, instances, x1Criterion, x2Criterion, classCriterion);

					double Px1x2Giveny = InstanceCounter.probabilityOfCriteriaGivenLabel
							(classLabel, y, instances, x1Criterion, x2Criterion);

					sum += Px1x2y * log2(Px1x2Giveny / (Px1Giveny * Px2Giveny));
				}
			}
		}
		return sum;
	}

	private static double log2(double a) {
		return StrictMath.log(a) / StrictMath.log(2.0);
	}

	public static List<List<Instance>> subsetsUsingStratifiedSampling(List<Instance> instances,
																	  int folds) {
		List<Instance> firstLabelInstances = new ArrayList<>();
		List<Instance> secondLabelInstances = new ArrayList<>();
		String firstClassLabel = instances.get(0).getClassLabel();
		for (Instance instance : instances) {
			if (!instance.getClassLabel().equals(firstClassLabel)) firstLabelInstances.add(instance);
			else secondLabelInstances.add(instance);
		}

		int rockSubsetSize = (int) Math.round((double) firstLabelInstances.size() / folds);
		int mineSubsetSize = (int) Math.round((double) secondLabelInstances.size() / folds);

		List<List<Instance>> subsets = new ArrayList<>();
		for (int foldIndex = 0; foldIndex < folds; foldIndex++) {
			List<Instance> sublist = new ArrayList<>();

			addInstancesFromCategory(folds, firstLabelInstances, rockSubsetSize, foldIndex, sublist);
			addInstancesFromCategory(folds, secondLabelInstances, mineSubsetSize, foldIndex, sublist);

			subsets.add(sublist);
		}

		return subsets;
	}

	private static void addInstancesFromCategory(int folds, List<Instance> firstInstances, int
			secondInstances, int foldIndex, List<Instance> sublist) {
		int subsetEnd, subsetStart = secondInstances * foldIndex;
		if (subsetStart < firstInstances.size()) {
			subsetEnd = Math.min(secondInstances * (foldIndex + 1), firstInstances.size());
			if (foldIndex == folds - 1) {
				subsetEnd = firstInstances.size();
			}
			if (subsetEnd <= firstInstances.size()) {
				sublist.addAll(firstInstances.subList(subsetStart, subsetEnd));
			}
		}
	}
}
