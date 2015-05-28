package glaux
package nn

import org.nd4j.api.linalg.DSL._
import org.nd4j.api.linalg.RichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import Dimension._


trait Vol {
  def indArray: INDArray
  type Dimensionality <: Dimension

  def dimension: Dimensionality
}

abstract class VolBase[DT <: Dimension: DimensionFactory](indArray: INDArray) extends Vol {
  type Dimensionality = DT

  val dimension: Dimensionality =
    implicitly[DimensionFactory[Dimensionality]].create(indArray.shape())
}


case class Vol3D(indArray: INDArray) extends VolBase[ThreeD](indArray) 

case class Matrix(indArray: INDArray) extends VolBase[TwoD](indArray)


case class RowVector(indArray: INDArray) extends VolBase[Row](indArray) {
  def sum = indArray.sum(Row.dimIndexOfData).getDouble(0)
}

object Vol3D {
  implicit def toVol(indArray: INDArray): Vol3D = Vol3D(indArray)

  val apply = Vol.applyBase[Vol3D]

}

object Matrix {
  implicit def toVol(indArray: INDArray): Matrix = Matrix(indArray)

  val apply = Vol.applyBase[Matrix]
}


object RowVector{
  implicit def toVol(indArray: INDArray): RowVector = RowVector(indArray)

  def apply(values: Double*): RowVector = RowVector(Dimension.Row(values.length), values)

  val apply = Vol.applyBase[RowVector]

}

object Vol {
  def apply(dimension: Dimension, data: Seq[Double]): Vol = toVol(createINDArray(dimension, data))

  def applyBase[V <: Vol] : (V#Dimensionality, Seq[Double]) => V = (dimension, data) =>
    apply(dimension, data).asInstanceOf[V]

  private def createINDArray(dimension: Dimension, data: Seq[Double]): INDArray = {
    assert(dimension.totalSize == data.length)
    Nd4j.create(data.toArray, dimension.shape)
  }


  implicit def toMatrix(vol: Vol) : INDArray = vol.indArray
  implicit def toRichMatrix(vol: Vol) : RichNDArray = toRichNDArray(vol.indArray)
  implicit def toVol(indArray: INDArray): Vol = Dimension.of(indArray) match {
    case d @ ThreeD(_,_,_) => Vol3D(indArray)
    case d @ TwoD(_,_) => Matrix(indArray)
    case d @ Row(_) => RowVector(indArray)
  }
}
