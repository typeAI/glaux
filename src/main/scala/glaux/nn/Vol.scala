package glaux
package nn

import org.nd4j.api.linalg.DSL._
import org.nd4j.api.linalg.RichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

trait Vol {
  def matrix: INDArray
  type Dimensionality <: Dimension

  def apply(indices: Int*) = matrix.getDouble(indices: _*)
  def dimension: Dimensionality
}

case class Vol3D(matrix: INDArray) extends Vol {
  type Dimensionality = Dimension3D
  lazy val dimension: Dimensionality = Dimension3D.of(matrix.shape())
}

case class Vol1D(matrix: INDArray) extends Vol {
  type Dimensionality = DimensionArray
  lazy val dimension: Dimensionality = DimensionArray(matrix.size(0))
  def sum = matrix.sum(DimensionArray.ndDimOfData).getDouble(0)
}

object Vol1D{
  def apply(values: Double*): Vol1D =  Vol1D(Nd4j.create(values.toArray, Array(values.length)))

  implicit def toVol(matrix: INDArray): Vol1D = Vol1D(matrix)
}

object Vol {
  def apply(dimension: Dimension, data: Double*): Vol = dimension match {
    case Dimension3D(x,y,z) => Vol3D(Nd4j.create(data.toArray, Array(x, y, z)))
    case DimensionArray(s) => Vol1D(data: _*)
  }


  implicit def toMatrix(vol: Vol) : INDArray = vol.matrix
  implicit def toRichMatrix(vol: Vol) : RichNDArray = toRichNDArray(vol.matrix)
  implicit def toVol(matrix: INDArray): Vol = Dimension.of(matrix) match {
    case d @ Dimension3D(_,_,_) => Vol3D(matrix)
    case d @ DimensionArray(_) => Vol1D(matrix)
  }
}

trait Dimension

object Dimension {
  def of(matrix: INDArray): Dimension = {
    val shape = matrix.shape()
    shape.length match {
      case 3 => Dimension3D.of(shape)
      case DimensionArray.numOfNDDims => DimensionArray(shape)
      case _ => throw new UnsupportedOperationException(s"unrecognized matrix shape $shape")
    }
  }
}
case class Dimension3D(x: Int, y: Int, z: Int) extends Dimension {
  assert(x > 0 && y > 0 && z > 0)
}

case class DimensionArray(size: Int) extends Dimension {
  assert(size > 0)
}

object DimensionArray {
  val ndDimOfData: Int = 1
  val numOfNDDims: Int = 2

  def apply(shape: Array[Int]): DimensionArray = DimensionArray(shape(ndDimOfData))
}



object Dimension3D {
  def of(shape: Array[Int]): Dimension3D = Dimension3D(shape(0), shape(1), shape(2))
}