package hab.cs760.bayesnet;

import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.List;

/**
 * Created by hannah on 11/5/17.
 */
public class InstanceCounter {

	/**
	 * Calculates P(x1, ..., xn | y) using Laplacian estimation. Basically count all instances
	 * with label y that satisfy the limiting values given as criteria, and divide by the total
	 * number of instances with label y, adding appropriate Laplacian pseudocounts to both
	 * numerator and denominator.
	 * @param classLabel definition of the given label
	 * @param classLabelValue given label value
	 * @param instances instances over which to count
	 * @param criteria the feature definitions and values for these features that should be counted.
	 * @return probability (between 0 and 1)
	 */
	public static double probabilityOfCriteriaGivenLabel(NominalFeature classLabel, String
			classLabelValue, List<Instance> instances, Criterion... criteria) {
		final int[] eligibleInstanceCount = {0};
		final int[] passingInstanceCount = {0};

		instances.forEach(instance -> {
			if (classLabel == null || instance.getFeatureValue(classLabel).equals
					(classLabelValue)) {
				eligibleInstanceCount[0]++;

				boolean allSatisfied = true;
				for (Criterion criterion : criteria) {
					if (!instance.getFeatureValue(criterion.limitingFeature).equals(criterion.limitingValue)) {
						allSatisfied = false;
					}
				}
				if (allSatisfied) {
					passingInstanceCount[0]++;
				}
			}
		});
		int denominatorPseudocount = 1;
		for (Criterion criterion : criteria) {
			denominatorPseudocount *= criterion.limitingFeature.possibleValues.size();
		}
		return (passingInstanceCount[0] + 1.0) / (eligibleInstanceCount[0] + denominatorPseudocount);
	}

	/**
	 * Calculates P(x0 | x1, ..., xn) using Laplacian estimation. Basically count all instances
	 * that satisfy x0 as well as x1...xn, and divide by the total number of instances satisfying
	 * x1...xn, adding appropriate Laplacian pseudocounts to both numerator and denominator.
	 * @param criterionToCount x0
	 * @param valueToCount value of x0
	 * @param instances instances over which to perform counting
	 * @param criteria x1...xn
	 * @return probability (between 0 and 1)
	 */
	public static double probabilityOfCriterionGivenCriteria(NominalFeature criterionToCount, String
			valueToCount, List<Instance> instances, Criterion... criteria) {
		final int[] eligibleInstanceCount = {0};
		final int[] passingInstanceCount = {0};

		instances.forEach(instance -> {
			boolean allSatisfied = true;
			for (Criterion criterion : criteria) {
				if (!instance.getFeatureValue(criterion.limitingFeature).equals(criterion.limitingValue)) {
					allSatisfied = false;
				}
			}
			if (allSatisfied) {
				eligibleInstanceCount[0]++;
				if (instance.getFeatureValue(criterionToCount).equals(valueToCount)) {
					passingInstanceCount[0]++;

				}
			}
		});
		int denominatorPseudocount = criterionToCount.possibleValues.size();
		return (passingInstanceCount[0] + 1.0) / (eligibleInstanceCount[0] + denominatorPseudocount);
	}


	public static class Criterion {
		public final NominalFeature limitingFeature;
		public final String limitingValue;

		public Criterion(NominalFeature limitingFeature, String limitingValue) {
			this.limitingFeature = limitingFeature;
			this.limitingValue = limitingValue;
		}
	}
}
