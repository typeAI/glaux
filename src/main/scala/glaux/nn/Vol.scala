package glaux
package nn

import org.nd4j.api.linalg.DSL._
import org.nd4j.api.linalg.RichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import Dimension._


trait Vol {
  def matrix: INDArray
  type Dimensionality <: Dimension

  def apply(indices: Int*): Double = matrix.getDouble(indices: _*)
  def dimension: Dimensionality
}

abstract class VolBase[DT <: Dimension: DimensionFactory](matrix: INDArray) extends Vol {
  type Dimensionality = DT

  val dimension: Dimensionality =
    implicitly[DimensionFactory[Dimensionality]].create(matrix.shape())
}

case class Vol3D(matrix: INDArray) extends VolBase[Dimension3D](matrix)

object Vol3D {
  implicit def toVol(matrix: INDArray): Vol3D = Vol3D(matrix)
}

case class Vol1D(matrix: INDArray) extends VolBase[DimensionArray](matrix) {
  def sum = matrix.sum(DimensionArray.dimIndexOfData).getDouble(0)
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
