package glaux

import glaux.statistics.impl.apache.ApacheImplementations

package object statistics {
  val distributions: DistributionImplementations = ApacheImplementations
}
