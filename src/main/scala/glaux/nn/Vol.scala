package glaux.nn

import org.nd4j.api.linalg.DSL._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

case class Vol private(matrix: INDArray) {
  lazy val dimension: Dimension = Dimension.of(matrix)
}


object Vol {
  def apply(data: Iterable[Float], dimension: Dimension): Vol =
    Vol(Nd4j.create(data.toArray, Array(dimension.x, dimension.y, dimension.z)))
}

case class Dimension(x: Int, y: Int, z: Int)
object Dimension {
  def of(matrix: INDArray): Dimension = {
    val shape = matrix.shape()
    Dimension(shape(0), shape(1), shape(2))
  }
}