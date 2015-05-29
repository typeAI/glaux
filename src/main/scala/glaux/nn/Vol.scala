package glaux
package nn

import glaux.nn.Vol.CanBuildFrom
import org.nd4j.api.linalg.DSL._
import org.nd4j.api.linalg.RichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import Dimension._


sealed trait Vol {
  def indArray: INDArray
  type Dimensionality <: Dimension
  def dimension: Dimensionality
}

class VolOps[V <: Vol: CanBuildFrom] {
  def map(v: V, f: Double => Double): V = {
    val dup = v.indArray.dup
    val linear = dup.linearView
    Range(0, linear.length).foreach { i =>
      linear.putScalar(i, f(linear.getDouble(i)))
    }
    dup
  }

  def map2(v1: V, v2: V, f: (Double, Double) => Double) : V = {
    assert(v1.dimension == v2.dimension)
    val dup = v1.indArray.dup
    val linear1 = dup.linearView
    val linear2 = v2.indArray.linearView
    Range(0, linear1.length).foreach { i =>
      linear1.putScalar(i, f(linear1.getDouble(i), linear2.getDouble(i)))
    }
    linear1
  }
}

sealed abstract class VolBase[DT <: Dimension: DimensionFactory](indArray: INDArray) extends Vol {
  type Dimensionality = DT

  val dimension: Dimensionality =
    implicitly[DimensionFactory[Dimensionality]].create(indArray.shape())
}

sealed abstract class VolCompanionBase[V <: Vol] {
  implicit val cb : CanBuildFrom[V]

  implicit lazy val ops = new VolOps[V]

  private def createINDArray(dimension: Dimension, data: Seq[Double]): INDArray = {
    assert(dimension.totalSize == data.length)
    Nd4j.create(data.toArray, dimension.shape)
  }

  def apply(dimension: Dimension, data: Seq[Double]): V = createINDArray(dimension, data)
}

case class Vol3D(indArray: INDArray) extends VolBase[ThreeD](indArray)

case class Matrix(indArray: INDArray) extends VolBase[TwoD](indArray)

case class RowVector(indArray: INDArray) extends VolBase[Row](indArray) {
  def sum = indArray.sum(Row.dimIndexOfData).getDouble(0)
}

object Vol3D extends VolCompanionBase[Vol3D] {
  implicit val cb : CanBuildFrom[Vol3D] = Vol3D.apply
}

object Matrix extends VolCompanionBase[Matrix]{
  implicit val cb : CanBuildFrom[Matrix] = Matrix.apply
  def apply(x: Int, y: Int, data: Seq[Double]): Matrix = apply(Dimension.TwoD(x,y), data)
}


object RowVector extends VolCompanionBase[RowVector]{
  implicit val cb : CanBuildFrom[RowVector] = RowVector.apply
  def apply(values: Double*): RowVector = RowVector(Dimension.Row(values.length), values)
}

object Vol extends VolCompanionBase[Vol]{
  implicit val cb : CanBuildFrom[Vol] = indArray => Dimension.of(indArray) match {
    case d @ ThreeD(_,_,_) => Vol3D(indArray)
    case d @ TwoD(_,_) => Matrix(indArray)
    case d @ Row(_) => RowVector(indArray)
  }

  type CanBuildFrom[T <: Vol] = INDArray => T

  implicit def toINDArray(vol: Vol) : INDArray = vol.indArray
  implicit def toRichIndArray(vol: Vol) : RichNDArray = toRichNDArray(vol.indArray)

}
