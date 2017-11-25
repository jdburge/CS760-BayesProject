package hab.cs760.test;

import hab.cs760.bayesnet.BayesNet;
import hab.cs760.bayesnet.FeatureNode;
import hab.cs760.bayesnet.LabelNode;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.Feature;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by hannah on 10/30/17.
 */
public class NaiveBayesTest2 {
	private List<Feature> features;
	private NominalFeature classLabel;
	private List<Instance> instances;


	@BeforeEach
	void setupStuff() {
		NominalFeature feature = new NominalFeature("lymphatics", 0, "normal", "arched",
				"deformed", "displaced");
		NominalFeature feature2 = new NominalFeature("by_pass", 1, "no", "yes");
		classLabel = new NominalFeature("class", 2, "metastases", "malign_lymph");
		features = Arrays.asList(feature, feature2, classLabel);

		Instance instance = new Instance();
		instance.addFeature(feature, "arched");
		instance.addFeature(feature2, "no");
		instance.addFeature(classLabel, "malign_lymph");

		instances = Collections.singletonList(instance);
	}

	@Test
	void testNaiveBayesStructure() {
		BayesNet naiveNet = BayesNet.naiveNet(features);

		assertEquals(classLabel, naiveNet.labelNode.feature);
		assertEquals(2, naiveNet.labelNode.connectedEdges.size());
		assertEquals(classLabel, ((FeatureNode)naiveNet.labelNode.connectedEdges.get(0).end()).
				labelNodeEdge.start().feature);
		assertEquals(classLabel, ((FeatureNode)naiveNet.labelNode.connectedEdges.get(1).end()).
				labelNodeEdge.start().feature);
	}

	@Test
	void testNaiveBayesLearn() {
		BayesNet naiveNet = BayesNet.naiveNet(features);
		naiveNet.train(instances);

		// test root node
		LabelNode labelNode = naiveNet.labelNode;
		assertEquals(1/3.0, labelNode.probabilityOf("metastases"));
		assertEquals(2/3.0, labelNode.probabilityOf("malign_lymph"));

		// test lymphatics node (left child)
		Node lymphatics = naiveNet.labelNode.connectedEdges.get(0).end();

		assertEquals(1/4.0, naiveBayesProb(lymphatics, "metastases", "normal"));
		assertEquals(1/4.0, naiveBayesProb(lymphatics, "metastases", "arched"));
		assertEquals(1/4.0, naiveBayesProb(lymphatics, "metastases", "deformed"));
		assertEquals(1/4.0, naiveBayesProb(lymphatics, "metastases", "displaced"));

		assertEquals(1/5.0, naiveBayesProb(lymphatics, "malign_lymph", "normal"));
		assertEquals(2/5.0, naiveBayesProb(lymphatics, "malign_lymph", "arched"));
		assertEquals(1/5.0, naiveBayesProb(lymphatics, "malign_lymph", "deformed"));
		assertEquals(1/5.0, naiveBayesProb(lymphatics, "malign_lymph", "displaced"));


		// test by_pass node (right child)
		Node byPass = naiveNet.labelNode.connectedEdges.get(1).end();

		assertEquals(1/2.0, naiveBayesProb(byPass, "metastases", "no"));
		assertEquals(1/2.0, naiveBayesProb(byPass, "metastases", "yes"));
		assertEquals(2/3.0, naiveBayesProb(byPass, "malign_lymph", "no"));
		assertEquals(1/3.0, naiveBayesProb(byPass, "malign_lymph", "yes"));
	}

	private double naiveBayesProb(Node node, String labelValue, String featureValue) {
		return node.probabilityOf(labelValue,null, featureValue);
	}

	@Test
	void testToString() {
		BayesNet naiveNet = BayesNet.naiveNet(features);
		naiveNet.train(instances);
		String expectedRepresentation = "lymphatics class\nby_pass class";
		assertEquals(expectedRepresentation, naiveNet.toString());
	}

}
