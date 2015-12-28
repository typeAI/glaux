package glaux.persistence.mongodb

import java.io.{ObjectOutputStream, ByteArrayOutputStream, ByteArrayInputStream, ObjectInputStream}
import java.time.{ZoneId, Instant, ZonedDateTime}

import reactivemongo.bson._

import scala.reflect.runtime.universe._

import play.api.libs.functional._

object GeneralHandlers {
  type Handler[T] = BSONDocumentReader[T] with BSONDocumentWriter[T]

  trait HandlerTrait[T] extends BSONDocumentReader[T] with BSONDocumentWriter[T]

  object HandlerTrait {
    def apply[T](reader: BSONDocumentReader[T], writer: BSONDocumentWriter[T]) = new HandlerTrait[T] {
      def write(t: T): BSONDocument = writer.write(t)
      def read(bson: BSONDocument): T = reader.read(bson)
    }
  }

  type Reader[T] = BSONReader[_ <: BSONValue, T]
  type Writer[T] = BSONWriter[T, _ <: BSONValue]

  implicit val fcbReader: FunctionalCanBuild[BSONDocumentReader] = new FunctionalCanBuild[BSONDocumentReader] {
    def apply[A, B](ma: BSONDocumentReader[A], mb: BSONDocumentReader[B]) = new BSONDocumentReader[~[A, B]] {
      def read(b: BSONDocument): ~[A, B] = new ~(ma.read(b), mb.read(b))
    }
  }

  implicit val fcbWriter: FunctionalCanBuild[BSONDocumentWriter] = new FunctionalCanBuild[BSONDocumentWriter] {
    def apply[A, B](ma: BSONDocumentWriter[A], mb: BSONDocumentWriter[B]) = new BSONDocumentWriter[~[A, B]] {
      def write(t: ~[A, B]): BSONDocument = t match { case (a ~ b) ⇒ ma.write(a) ++ mb.write(b) }
    }
  }

  implicit val fcbHandler: FunctionalCanBuild[Handler] = new FunctionalCanBuild[Handler] {
    def apply[A, B](fa: Handler[A], fb: Handler[B]): Handler[A ~ B] = HandlerTrait[A ~ B](fcbReader(fa, fb), fcbWriter(fa, fb))
  }

  implicit val invariantFunctorHandler: InvariantFunctor[Handler] = new InvariantFunctor[Handler] {
    def inmap[A, B](m: Handler[A], f1: (A) ⇒ B, f2: (B) ⇒ A): Handler[B] = new HandlerTrait[B] {
      def write(b: B): BSONDocument = m.write(f2(b))
      def read(bson: BSONDocument): B = f1(m.read(bson))
    }
  }

  def field[T](fieldName: String)(implicit reader: BSONReader[_ <: BSONValue, T], writer: BSONWriter[T, _ <: BSONValue]): Handler[T] = new HandlerTrait[T] {
    def read(bson: BSONDocument): T = bson.getAs[T](fieldName).get
    def write(t: T): BSONDocument = BSONDocument(fieldName → t)
  }

  def nullableField[T](fieldName: String)(implicit reader: BSONReader[_ <: BSONValue, T], writer: BSONWriter[T, _ <: BSONValue]): Handler[Option[T]] = new HandlerTrait[Option[T]] {
    def read(bson: BSONDocument): Option[T] = bson.getAs[T](fieldName)
    def write(t: Option[T]): BSONDocument = t.map(f ⇒ BSONDocument(fieldName → f)).getOrElse(BSONDocument.empty)
  }

  implicit class HandlerOpts[T](self: Handler[T]) {
    def cast[To]: Handler[To] = invariantFunctorHandler.inmap(self, (d: T) ⇒ d.asInstanceOf[To], (d: To) ⇒ d.asInstanceOf[T])
  }

  implicit def tuple2F[T1: Reader: Writer, T2: Reader: Writer] = new HandlerTrait[(T1, T2)] {
    def read(bson: BSONDocument): (T1, T2) = (bson.getAs[T1]("_1").get, bson.getAs[T2]("_2").get)

    def write(t: (T1, T2)): BSONDocument = BSONDocument(
      "_1" → t._1,
      "_2" → t._2
    )
  }

  implicit def tuple3F[T1: Reader: Writer, T2: Reader: Writer, T3: Reader: Writer] = new HandlerTrait[(T1, T2, T3)] {
    def read(bson: BSONDocument): (T1, T2, T3) = (
      bson.getAs[T1]("_1").get,
      bson.getAs[T2]("_2").get,
      bson.getAs[T3]("_3").get
    )

    def write(t: (T1, T2, T3)): BSONDocument = BSONDocument(
      "_1" → t._1,
      "_2" → t._2,
      "_3" → t._3
    )
  }

  def binaryFormat[T]: Handler[T] = new HandlerTrait[T] {
    def read(bson: BSONDocument): T = {
      val array = bson.get("binary").get.asInstanceOf[BSONBinary].byteArray
      val r = new ObjectInputStream(new ByteArrayInputStream(array)).readObject()
      r.asInstanceOf[T]
    }

    def write(t: T): BSONDocument = {
      val bos = new ByteArrayOutputStream()
      val oos = new ObjectOutputStream(bos)
      try {
        oos.writeObject(t)
      } catch {
        case e: Exception ⇒
          e.printStackTrace()
          throw e
      }

      oos.close

      BSONDocument(
        "class" → t.getClass.getCanonicalName,
        "binary" → BSONBinary(bos.toByteArray, Subtype.GenericBinarySubtype)
      )
    }
  }

  implicit val zdtf = new HandlerTrait[ZonedDateTime] {
    def read(bson: BSONDocument): ZonedDateTime = ZonedDateTime.ofInstant(
      Instant.ofEpochMilli(bson.getAs[BSONDateTime]("epoch").get.value),
      ZoneId.of(bson.getAs[String]("zone").get)
    )

    def write(t: ZonedDateTime): BSONDocument = BSONDocument(
      "zone" → t.getZone.getId,
      "epoch" → BSONDateTime(t.toInstant.toEpochMilli)
    )
  }

  class Partial[ParentT, ChildT <: ParentT: Handler: TypeTag] {
    val typeTag = implicitly[TypeTag[ChildT]]
    val runtimeClass = typeTag.mirror.runtimeClass(typeTag.tpe)
    def name: String = {
      typeTag.tpe.typeSymbol.fullName
    }
    def read(bson: BSONDocument): ParentT = bson.as[ChildT]
    def write(t: ParentT) = BSON.write(t.asInstanceOf[ChildT])
    def isDefined(t: ParentT): Boolean = runtimeClass.isInstance(t)
  }

  /**
   * Use partial function rather than runtimeClass which is prone to erasure.
   */
  class GPartial[ParentT, ChildT <: ParentT: Handler: TypeTag](pf: PartialFunction[ParentT, Unit]) extends Partial[ParentT, ChildT] {
    override def isDefined(t: ParentT): Boolean = pf.isDefinedAt(t)
  }

  def polymorphic[T](partials: Partial[T, _]*) = new HandlerTrait[T] {

    def read(bson: BSONDocument): T = {
      val p = partials.find(_.name == bson.getAs[String]("type").get).get
      p.read(bson.get("item").get.asInstanceOf[BSONDocument])
    }

    def write(t: T): BSONDocument = {
      val p = partials.find(_.isDefined(t)).get
      BSONDocument("type" → p.name, "item" → p.write(t))
    }
  }

  implicit val unitHandler: Handler[Unit] = new HandlerTrait[Unit] {
    override def write(t: Unit): BSONDocument = BSONDocument.empty

    override def read(bson: BSONDocument): Unit = ()
  }
}
