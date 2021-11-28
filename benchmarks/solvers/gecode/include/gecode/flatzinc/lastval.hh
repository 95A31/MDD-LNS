/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *     Jip J. Dekker <jip.dekker@monash.edu>
 *
 *  Copyright:
 *     Jip J. Dekker, 2018
 *
 *  This file is part of Gecode, the generic constraint
 *  development environment:
 *     http://www.gecode.org
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __FLATZINC_LASTVAL_HH__
#define __FLATZINC_LASTVAL_HH__

#include <gecode/int.hh>
#include <memory>
using namespace Gecode::Int;

namespace Gecode { namespace FlatZinc {

  class LastVal : public UnaryPropagator<IntView, PC_INT_VAL> {
  protected:
    using UnaryPropagator<IntView, PC_INT_VAL>::x0;
    std::shared_ptr<int> s;

    /// Constructor for cloning \a p
    LastVal(Space& home, LastVal& p);
    /// Constructor for posting
    LastVal(Home home, Int::IntView x0, std::shared_ptr<int> s);
  public:
    /// Copy propagator during cloning
    virtual Actor* copy(Space& home);
    /// Cost function (defined as TODO)
    virtual PropCost cost(const Space& home, const ModEventDelta& med) const;
    /// Perform propagation
    virtual ExecStatus propagate(Space& home, const ModEventDelta& med);

    static ExecStatus post(Home home, Int::IntView x0, std::shared_ptr<int> s);
  };

}}

#endif //__FLATZINC_LASTVAL_HH__
