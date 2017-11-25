package hab.cs760.bayesnet;

import java.util.Arrays;
import java.util.List;

/**
 * Created by hannah on 11/5/17.
 */
public class Edge {
	public final Node node1;
	public final Node node2;
	public double weight;
	private Direction direction;
	private final Direction forwardDirection = new Direction() {
		@Override
		public Node start() {
			return node1;
		}

		@Override
		public Node end() {
			return node2;
		}
	};
	private final Direction backwardDirection = new Direction() {

		@Override
		public Node start() {
			return node2;
		}

		@Override
		public Node end() {
			return node1;
		}
	};

	private Edge(Node node1, Node node2) {
		this.node1 = node1;
		this.node2 = node2;
		this.direction = null;
	}

	public List<Node> connectedNodes() {
		return Arrays.asList(node1, node2);
	}

	public static Edge directedEdge(Node start, Node end) {
		Edge edge = new Edge(start, end);
		edge.direction = edge.forwardDirection;
		return edge;
	}

	public static Edge undirectedEdge(Node edge1, Node edge2) {
		return new Edge(edge1, edge2);
	}

	public void setStart(Node start) {
		if (node1 != start && node2 != start) throw new IllegalStateException("Given node is not " +
				"attached to this edge");
		if (start == node1) {
			direction = forwardDirection;
		} else {
			direction = backwardDirection;
		}
	}

	interface Direction {
		Node start();
		Node end();
	}

	public Node start() throws UndirectedEdgeException {
		if (direction == null) {
			throw new UndirectedEdgeException();
		} else {
			return direction.start();
		}
	}
	public Node end() throws UndirectedEdgeException {
		if (direction == null) {
			throw new UndirectedEdgeException();
		} else {
			return direction.end();
		}
	}

	boolean isUndirected() {
		return direction == null;
	}

	class UndirectedEdgeException extends RuntimeException {
		UndirectedEdgeException() {
			super("Start and end points are undefined for this edge");
		}
	}

	@Override
	public String toString() {
		if (direction != null) return start().feature.name + " ---> " + end()
				.feature.name;
		else return node1.feature.name + " ---- " + node2.feature.name;
	}
}
