package glaux
package nn

import glaux.nn.Vol.CanBuildFrom
import org.nd4j.api.linalg.DSL._
import org.nd4j.api.linalg.RichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import Dimension._


trait Vol {
  def indArray: INDArray
  type Dimensionality <: Dimension
  def dimension: Dimensionality
  def sumAll: Double = { indArray.linearView().sum(Row.dimIndexOfData).getDouble(0) }
}

sealed abstract class VolBase[D <: Dimension : DimensionFactory](indArray: INDArray) extends Vol {
  type Dimensionality = D
  val dimension: Dimensionality =
    implicitly[DimensionFactory[Dimensionality]].create(indArray.shape())

}

case class Vol3D(indArray: INDArray)      extends VolBase[ThreeD](indArray)
case class Matrix(indArray: INDArray)     extends VolBase[TwoD](indArray)
case class RowVector(indArray: INDArray)  extends VolBase[Row](indArray) {
  def apply(index: Int) : Double = indArray.getDouble(index)
}


abstract class VolCompanionBase[V <: Vol] {
  implicit val cb : CanBuildFrom[V]

  private def createINDArray(dimension: Dimension, data: Seq[Double]): INDArray = {
    assert(dimension.totalSize == data.length, s"data length ${data.length} does not conform to $dimension" )
    Nd4j.create(data.toArray, dimension.shape)
  }

  def apply(dimension: Dimension, data: Seq[Double]): V = createINDArray(dimension, data)

  def uniform(dimension: Dimension, value: Double): V = Nd4j.create(dimension.shape:_*).assign(value)
}

object RowVector extends VolCompanionBase[RowVector]{
  implicit val cb : CanBuildFrom[RowVector] = RowVector.apply
  def apply(values: Double*): RowVector = RowVector(Dimension.Row(values.length), values)
}

object Vol3D extends VolCompanionBase[Vol3D] {
  implicit val cb : CanBuildFrom[Vol3D] = Vol3D.apply
  def apply(x: Int, y: Int, z: Int, data: Seq[Double]): Vol3D = apply(Dimension.ThreeD(x,y,z), data)
}

object Matrix extends VolCompanionBase[Matrix]{
  implicit val cb : CanBuildFrom[Matrix] = Matrix.apply
  def apply(x: Int, y: Int, data: Seq[Double]): Matrix = apply(Dimension.TwoD(x,y), data)
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


  implicit class VolOps[V <: Vol: CanBuildFrom](v: V) {

    def map(f: Double => Double): V = mapWithIndex((value, i) => f(value))

    def mapWithIndex(f: (Double, Int) => Double): V = {
      val dup = v.indArray.dup
      val linear = dup.linearView
      Range(0, linear.length).foreach { i =>
        linear.putScalar(i, f(linear.getDouble(i), i))
      }
      dup
    }

    def merge(v2: V)(f: (Double, Double) => Double) : V = {
      assert(v.dimension == v2.dimension)
      val linear2 = v2.indArray.linearView
      mapWithIndex((value, i) => f(value, linear2.getDouble(i) ))
    }
  }

}
