package glaux.linalg

import glaux.linalg.Dimension._
import glaux.linalg.Vol._

trait Vol extends VolOperations {

  type Dimensionality <: Dimension
  def dimension: Dimensionality
  def sumAll: Double = iterable.sum

  def iterable: Iterable[Double]
  def toArray: Array[Double] = iterable.toArray
}


trait VolOperations {

  def +(that: Vol): Vol

  def -(that: Vol): Vol

  /** element-by-element multiplication */
  def *(that: Vol): Vol

  /** matrix multiplication */
  def **(that: Vol): Vol

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: Vol): Vol

  def /(that: Vol): Vol

  /** right division ... is this the correct symbol? */
  def \(that: Vol): Vol

  def +(that: Number): Vol

  def -(that: Number): Vol

  def *(that: Number): Vol

  def /(that: Number): Vol

  def \(that: Number): Vol

  def T: Vol

}

object Vol {
  
  type CanBuildFrom[From, V <: Vol] = From => V
  type VolBuilder[V <: Vol] = CanBuildFrom[(V#Dimensionality, Iterable[Double]), V]
  type GenVolBuilder[V <: Vol] = CanBuildFrom[(Dimension, Iterable[Double]), V]
  type RowBuilder = VolBuilder[RowVector]
  type MatrixBuilder = VolBuilder[Matrix]
  type Vol3DBuilder = VolBuilder[Vol3D]


  implicit val toRow: CanBuildFrom[Vol, RowVector] = v => v.asInstanceOf[RowVector]
  implicit val toMatrix: CanBuildFrom[Vol, Matrix] = v => v.asInstanceOf[Matrix]

  implicit def toGen[V <: Vol](implicit gb: GenVolBuilder[V]): VolBuilder[V] = gb

  implicit class VolOps[V <: Vol : VolBuilder](v: V) {

    def map(f: Double => Double): V = (v.dimension, v.iterable.map(f))

    def merge(v2: V)(f: (Double, Double) => Double) : V = {
      assert(v.dimension == v2.dimension)
      (v.dimension, v.iterable.zip(v2.iterable).map(f.tupled))
    }
  }



}

trait RowVector extends Vol {
  type Dimensionality = Row
  def apply(index: Int) : Double
}

trait Matrix extends Vol {
  type Dimensionality = TwoD
}

trait Vol3D extends Vol {
  type Dimensionality = ThreeD
}


trait VolFactory[V <: Vol] {
  def apply(dimension: V#Dimensionality, data: Iterable[Double])(implicit b: VolBuilder[V]) : V = b((dimension, data))

  def uniform(dimension: V#Dimensionality, value: Double)(implicit b: VolBuilder[V]): V = apply(dimension, Array.fill(dimension.totalSize)(value))
//
//  def normal(dimension: Dimension, mean: Double, std: Double): V =
//    Nd4j.getDistributions.createNormal(mean, std).sample(dimension.shape)
}

object RowVector extends VolFactory[RowVector]{
  def apply(values: Double*)(implicit b: RowBuilder): RowVector = apply(Dimension.Row(values.length), values)
}

object Vol3D extends VolFactory[Vol3D] {
  def apply(x: Int, y: Int, z: Int, data: Seq[Double])(implicit b: Vol3DBuilder): Vol3D = apply(Dimension.ThreeD(x,y,z), data)
}

object Matrix extends VolFactory[Matrix]{
  def apply(x: Int, y: Int, data: Seq[Double])(implicit b: MatrixBuilder): Matrix = apply(Dimension.TwoD(x,y), data)
}
